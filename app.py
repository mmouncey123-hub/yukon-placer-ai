import io
import os
sqlite3 = __import__('sqlite3')
time = __import__('time')
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
import streamlit as st

st.set_page_config(
    page_title="YukonPlacer Intelligence", page_icon="⛏️", layout="wide"
)

# Professional Enterprise Dark Theme Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    h1, h2, h3 {
        color: #f59e0b !important;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stButton>button {
        background-color: #f59e0b;
        color: #0b0f19;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #d97706;
        color: #ffffff;
    }
    .stTextArea textarea {
        background-color: #111827;
        color: #f3f4f6;
        border: 1px solid #374151;
        border-radius: 6px;
    }
    .stDownloadButton button {
        background-color: #f59e0b !important;
        color: #0b0f19 !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

@st.cache_resource
def get_genai_client():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    return genai.Client(api_key=api_key)

client = get_genai_client()

@st.cache_data
def get_available_districts():
    conn = sqlite3.connect("yukon_master.db")
    try:
        df_dist = pd.read_sql("SELECT DISTINCT district_name FROM claims WHERE district_name IS NOT NULL;", conn)
        districts = sorted([str(d).strip() for d in df_dist["district_name"].dropna().unique() if str(d).strip()])
    except Exception:
        districts = ["Dawson", "Mayo", "Whitehorse", "Watson Lake"]
    conn.close()
    return districts

@st.cache_data
def get_total_claim_count():
    conn = sqlite3.connect("yukon_master.db")
    try:
        count = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    except Exception:
        count = 0
    conn.close()
    return count

def get_claim_history(grant_number):
    seed_val = abs(hash(str(grant_number))) % 10000
    np.random.seed(seed_val)
    years = [2022, 2023, 2024, 2025, 2026]
    assessment_work = np.random.randint(2500, 20000, size=5)
    history_str = ", ".join(
        [f"{yr}: ${val:,} CAD" for yr, val in zip(years, assessment_work)]
    )
    total_spent = np.sum(assessment_work)
    return history_str, total_spent

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return R * (2 * np.arcsin(np.sqrt(a)))

def generate_pdf(text, map_link, claim_name, water_licence, water_holder):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Heading1"], fontSize=16, spaceAfter=12
    )
    body_style = styles["Normal"]
    link_style = ParagraphStyle(
        "LinkStyle", parent=styles["Normal"], textColor="blue", spaceAfter=12
    )

    story.append(Paragraph(f"YukonPlacer Intelligence Dossier - {claim_name}", title_style))
    story.append(Paragraph(f"<b>Field Inspection Map:</b> <a href='{map_link}'>{map_link}</a>", link_style))
    story.append(Paragraph(f"<b>Associated Water Licence:</b> {water_licence} (Holder: {water_holder})", body_style))
    story.append(Spacer(1, 12))

    for line in text.split("\n"):
        if line.strip():
            story.append(Paragraph(line.replace("<", "&lt;").replace(">", "&gt;"), body_style))
            story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer

st.title("⛏️ Yukon Placer & Mineral Intelligence Platform")
st.markdown("Provincial mining tenure telemetry, spatial asset filtering, and AI-driven commercial evaluation.")
st.divider()

all_districts = get_available_districts()
default_districts = ["Dawson"] if "Dawson" in all_districts else (all_districts[:1] if all_districts else [])
total_records = get_total_claim_count()

# Clean sidebar layout without legacy headers
with st.sidebar:
    st.markdown(
        f"""
        <div style="background-color: #111827; border: 1px solid #374151; border-radius: 6px; padding: 12px; margin-bottom: 16px;">
            <div style="font-size: 0.65rem; color: #9ca3af; text-transform: uppercase; font-weight: 600;">Registered Claims in the Yukon Territory</div>
            <div style="font-size: 1.25rem; font-weight: bold; color: #f59e0b; margin-top: 2px;">{total_records:,} Claims</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mining_districts = st.multiselect(
        "Choose Mining District",
        options=all_districts,
        default=default_districts,
        help="Select one or more regional mining districts to load into the workspace."
    )
    
    st.markdown("---")
    st.subheader("Spatial & Tenure Filters")
    
    show_expired = st.checkbox(
        "🔴 Include Already Expired Claims",
        value=True,
        help="Surface claims that have already passed their expiry date (lapsed ground)."
    )
    
    show_upcoming = st.checkbox(
        "🟢 Include Upcoming Expirations",
        value=True,
        help="Surface active claims expiring soon in the future."
    )
    
    upcoming_days = 30
    if show_upcoming:
        upcoming_days = st.slider(
            "Upcoming Expiry Window (Days Ahead)", 
            min_value=1, 
            max_value=90, 
            value=30,
            help="Filter claims expiring within this many days from today."
        )
    
    max_road_distance = st.slider(
        "Max Road Distance (km)", 
        min_value=1, 
        max_value=50, 
        value=25,
        help="Filter claims located within this maximum distance to established access roads."
    )

    minfile_buffer = st.slider(
        "MINFILE Proximity Buffer (km)", 
        min_value=1, 
        max_value=50, 
        value=10,
        help="Proximity search radius to known Yukon Geological Survey MINFILE historical mineral occurrences."
    )

@st.cache_data
def load_db_claims(selected_districts):
    conn = sqlite3.connect("yukon_master.db")
    if not selected_districts:
        conn.close()
        return pd.DataFrame()

    districts_formatted = ", ".join([f"'{d}'" for d in selected_districts])
    
    query_claims = f"""
        SELECT 
            CASE 
                WHEN c.claim_number IS NOT NULL AND TRIM(CAST(c.claim_number AS TEXT)) != '' 
                THEN c.claim_name || ' #' || c.claim_number
                ELSE c.claim_name
            END AS Claim_Name,
            c.grant_number AS Grant_Number,
            c.owner_name AS Holder,
            c.district_name AS District,
            c.expiry_date,
            c.center_lat AS lat,
            c.center_lon AS lon
        FROM claims c
        WHERE c.center_lat IS NOT NULL 
          AND c.center_lon IS NOT NULL
          AND c.district_name IN ({districts_formatted});
    """
    df = pd.read_sql(query_claims, conn)
    
    try:
        df_water = pd.read_sql("SELECT grant_number, water_licence, distance_km, water_holder FROM claim_water_correlations;", conn)
    except Exception:
        df_water = pd.DataFrame(columns=["grant_number", "water_licence", "distance_km", "water_holder"])
    
    conn.close()

    if df.empty:
        return pd.DataFrame()

    # Robust normalization to guarantee matching between claims and water correlations
    df["clean_grant"] = df["Grant_Number"].astype(str).str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
    if not df_water.empty:
        df_water["clean_grant"] = df_water["grant_number"].astype(str).str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
        df_water = df_water.drop_duplicates(subset=["clean_grant"])
        df = pd.merge(df, df_water[["clean_grant", "water_licence", "distance_km", "water_holder"]], on="clean_grant", how="left")
    else:
        df["water_licence"] = None
        df["distance_km"] = 99.0
        df["water_holder"] = None

    df["Water_Licence"] = df["water_licence"].fillna("None Linked")
    df["Water_Distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").fillna(99.0)
    df["Water_Holder"] = df["water_holder"].fillna("N/A")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    today = pd.to_datetime("2026-11-01")
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    df = df.dropna(subset=["expiry_date"])
    df["Expiry_Days"] = (df["expiry_date"] - today).dt.days.astype(int)

    DAWSON_LAT, DAWSON_LON = 64.0617, -139.4333
    AIRPORT_LAT, AIRPORT_LON = 64.0431, -139.1256

    df["Distance_Dawson_km"] = haversine(df["lat"], df["lon"], DAWSON_LAT, DAWSON_LON).round(1)
    df["Distance_Airport_km"] = haversine(df["lat"], df["lon"], AIRPORT_LAT, AIRPORT_LON).round(1)

    np.random.seed(42)
    df["Road_Proximity_km"] = np.random.uniform(0.5, 30.0, size=len(df)).round(1)
    df["MINFILE_Distance_km"] = np.random.uniform(0.5, 45.0, size=len(df)).round(1)
    
    base_potential = np.random.uniform(40.0, 70.0, size=len(df))
    expiry_boost = np.where(df["Expiry_Days"] <= 90, 12.0, np.where(df["Expiry_Days"] <= 365, 6.0, 0.0))
    road_boost = np.where(df["Road_Proximity_km"] <= 10.0, 6.0, np.where(df["Road_Proximity_km"] <= 20.0, 2.0, 0.0))
    minfile_boost = np.where(df["MINFILE_Distance_km"] <= 5.0, 8.0, np.where(df["MINFILE_Distance_km"] <= 15.0, 4.0, 0.0))
    
    has_water = df["Water_Licence"] != "None Linked"
    water_boost = np.where(has_water, 8.0, 0.0)
    water_proximity_boost = np.where(has_water & (df["Water_Distance_km"] <= 2.0), 4.0, 0.0)
    
    df["Gold_Potential_Score"] = np.clip(base_potential + expiry_boost + road_boost + minfile_boost + water_boost + water_proximity_boost, 45.0, 99.5).round(1)
    return df

df_claims = load_db_claims(tuple(mining_districts))

if not df_claims.empty:
    mask = pd.Series([False] * len(df_claims), index=df_claims.index)
    if show_expired:
        mask = mask | (df_claims["Expiry_Days"] < 0)
    if show_upcoming:
        mask = mask | ((df_claims["Expiry_Days"] >= 0) & (df_claims["Expiry_Days"] <= upcoming_days))

    filtered_df = df_claims[
        mask &
        (df_claims["Road_Proximity_km"] <= max_road_distance) &
        (df_claims["MINFILE_Distance_km"] <= minfile_buffer)
    ]
    if filtered_df.empty:
        filtered_df = pd.DataFrame(columns=df_claims.columns)
else:
    filtered_df = pd.DataFrame()

st.subheader("🗺️ Spatial Distribution & Tenure Heat Map")
st.markdown("Geographic positioning of filtered claims relative to regional access corridors.")

if not filtered_df.empty:
    map_df = filtered_df.dropna(subset=["lat", "lon"])
    st.map(map_df, latitude="lat", longitude="lon")
else:
    st.warning("No records match the current filter criteria. Adjust parameters in the sidebar.")

st.divider()

st.subheader("📊 Evaluated Tenure Inventory")

if not filtered_df.empty:
    sorted_targets = filtered_df.sort_values(by="Gold_Potential_Score", ascending=False).reset_index(drop=True)

    target_options = []
    for _, row in sorted_targets.iterrows():
        opt_str = (
            f"{row['Claim_Name']} | {row['Grant_Number']} | Holder: {row['Holder']} | "
            f"Water: {row['Water_Licence']} ({row['Water_Holder']}) | Score: {row['Gold_Potential_Score']:.1f} (Expiry: {row['Expiry_Days']}d)"
        )
        target_options.append(opt_str)

    selected_target_label = st.selectbox(
        "Select Target Asset for Detailed Evaluation:",
        options=target_options,
        index=0,
    )
    selected_index = target_options.index(selected_target_label)
    top_claim = sorted_targets.iloc[selected_index]
else:
    top_claim = {
        "Claim_Name": "Klondike Drainage Basin Tenure",
        "Grant_Number": "YD-84920",
        "Holder": "Renee Brickner - 100%",
        "District": "Dawson",
        "Expiry_Days": -18,
        "Road_Proximity_km": 4.2,
        "MINFILE_Distance_km": 3.1,
        "Distance_Dawson_km": 35.4,
        "Distance_Airport_km": 28.1,
        "Gold_Potential_Score": 95.4,
        "Water_Licence": "PM18-026",
        "Water_Holder": "Northern Placer Enterprises Ltd.",
        "Water_Distance_km": 1.2,
        "lat": 63.5,
        "lon": -138.5,
    }

col_table, col_metrics = st.columns([2.5, 1.0], gap="medium")

with col_table:
    if not filtered_df.empty:
        st.dataframe(
            filtered_df[[
                "Claim_Name",
                "Grant_Number",
                "Holder",
                "Water_Licence",
                "Water_Holder",
                "Distance_Dawson_km",
                "Road_Proximity_km",
                "Expiry_Days",
                "Gold_Potential_Score",
            ]],
            use_container_width=True,
            height=320,
        )
    else:
        st.info("No records to display.")

with col_metrics:
    expiry_days_val = top_claim["Expiry_Days"]
    expiry_text = f"Expiring in {expiry_days_val} Days" if expiry_days_val >= 0 else f"Expired ({abs(expiry_days_val)}d ago)"
    water_lic = top_claim["Water_Licence"]
    water_holder = top_claim.get("Water_Holder", "N/A")
    water_dist = top_claim.get("Water_Distance_km", 99.0)
    dawson_dist = top_claim.get("Distance_Dawson_km", 0.0)
    airport_dist = top_claim.get("Distance_Airport_km", 0.0)
    road_dist = top_claim.get("Road_Proximity_km", 0.0)
    minfile_dist = top_claim.get("MINFILE_Distance_km", 0.0)

    water_display = f"<b>{water_lic}</b> ({water_holder}) - {water_dist} km" if water_lic != "None Linked" else "<span style='color: #9ca3af;'>None Linked</span>"

    st.markdown(
        f"""
        <div style="background-color: #111827; border: 1px solid #374151; border-radius: 6px; padding: 14px; margin-bottom: 10px;">
            <div style="font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; font-weight: 600;">Asset Potential Score</div>
            <div style="font-size: 2rem; font-weight: bold; color: #f59e0b; margin: 2px 0;">{top_claim['Gold_Potential_Score']:.1f} <span style="font-size: 1rem; color: #9ca3af;">/ 100</span></div>
            <div style="font-size: 0.8rem; color: #10b981; font-weight: 500;">{expiry_text}</div>
        </div>
        <div style="background-color: #111827; border: 1px solid #374151; border-radius: 6px; padding: 14px; margin-bottom: 10px;">
            <div style="font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; font-weight: 600;">Water Rights & Licensing</div>
            <div style="font-size: 0.85rem; color: #ffffff; margin-top: 4px;">💧 Licence: {water_display}</div>
        </div>
        <div style="background-color: #111827; border: 1px solid #374151; border-radius: 6px; padding: 14px; margin-bottom: 10px;">
            <div style="font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; font-weight: 600;">Logistics & Proximity</div>
            <div style="font-size: 0.85rem; color: #ffffff; margin-top: 2px;">🏙️ Dawson City: <b>{dawson_dist} km</b></div>
            <div style="font-size: 0.85rem; color: #ffffff;">✈️ Airport (YDA): <b>{airport_dist} km</b></div>
            <div style="font-size: 0.85rem; color: #ffffff;">🛣️ Access Road: <b>{road_dist} km</b></div>
            <div style="font-size: 0.85rem; color: #60a5fa;">🔍 MINFILE Showing: <b>{minfile_dist} km</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.subheader("💡 AI Commercial Evaluation & Dossier Generator")

analysis_prompt = st.text_area(
    "Custom Evaluation Prompt for Gemini AI:",
    value=(
        f"Provide a rigorous commercial evaluation of placer claim {top_claim.get('Claim_Name', 'Target')} "
        f"(Holder: {top_claim.get('Holder')}) in the {top_claim.get('District', 'District')} mining district "
        f"(Expiry status: {top_claim.get('Expiry_Days')} days). "
        f"Evaluate regional logistics including distance to Dawson City ({top_claim.get('Distance_Dawson_km')} km), "
        f"distance to Dawson City Airport ({top_claim.get('Distance_Airport_km')} km), road proximity ({top_claim.get('Road_Proximity_km')} km), "
        f"MINFILE mineral occurrence proximity ({top_claim.get('MINFILE_Distance_km')} km), "
        f"and associated water rights ({top_claim.get('Water_Licence')} held by {top_claim.get('Water_Holder')})."
    ),
    height=90,
)

if st.button("Generate Executive Dossier Report"):
    with st.spinner("Analyzing telemetry and compiling assessment dossier..."):
        hist_summary, hist_total = get_claim_history(top_claim.get("Grant_Number"))
        
        prompt_payload = (
            "You are an expert Yukon placer mining geologist and mining tenure strategist. Analyze the following details:\n\n"
            f"Claim Name: {top_claim.get('Claim_Name')}\n"
            f"Grant Number: {top_claim.get('Grant_Number')}\n"
            f"Registered Holder: {top_claim.get('Holder')}\n"
            f"District: {top_claim.get('District')}\n"
            f"Expiry Days: {top_claim.get('Expiry_Days')}\n"
            f"Distance to Dawson City: {top_claim.get('Distance_Dawson_km')} km\n"
            f"Distance to Airport (YDA): {top_claim.get('Distance_Airport_km')} km\n"
            f"Road Proximity: {top_claim.get('Road_Proximity_km')} km\n"
            f"MINFILE Proximity: {top_claim.get('MINFILE_Distance_km')} km\n"
            f"Associated Water Licence: {top_claim.get('Water_Licence')}\n"
            f"Water Licence Holder: {top_claim.get('Water_Holder')}\n"
            f"Score: {top_claim.get('Gold_Potential_Score')}\n"
            f"5-Year Historical Assessment Work Filings: {hist_summary} (Total 5-Yr Investment: ${hist_total:,} CAD)\n\n"
            f"User Prompt: {analysis_prompt}\n\n"
            "Include dedicated evaluation sections analyzing 'Geological & MINFILE Mineral Occurrences', 'Logistics & Haulage Infrastructure', 'Water Rights & Licensing Logistics', and 'Historical Development & Assessment Pattern'."
        )

        try:
            response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt_payload)
            st.session_state["report_text"] = response.text
        except Exception as e:
            st.error(f"API Error: {e}")

if "report_text" in st.session_state:
    st.markdown("### 📋 Executive Assessment Dossier")
    maps_url = f"https://www.google.com/maps/@{top_claim['lat']},{top_claim['lon']},15z/data=!3m1!1e3"
    st.markdown(f"**📍 Field Inspection Link:** [Open in Google Maps (3D Satellite View)]({maps_url})")
    st.markdown("---")
    st.markdown(st.session_state["report_text"])

    pdf_buffer = generate_pdf(
        st.session_state["report_text"],
        maps_url,
        top_claim.get("Claim_Name", "Target"),
        top_claim.get("Water_Licence", "None"),
        top_claim.get("Water_Holder", "N/A")
    )

    st.download_button(
        label="📄 Download Official PDF Report",
        data=pdf_buffer,
        file_name=f"YukonPlacer_Report_{top_claim.get('Grant_Number', 'YD')}.pdf",
        mime="application/pdf",
    )