import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import plotly.express as px

# -----------------------------------------------------------------------------
# Configuration & Layout
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Real-Time Earthquake Tracker & Risk Predictor",
    page_icon="🌋",
    layout="wide"
)

st.title("🌋 Real-Time Earthquake Monitor & Educational Predictor")
st.markdown("""
This application fetches live global data from the **USGS API** to help students analyze earthquake frequencies, 
magnitudes, and geographic patterns, making evacuation drills more interactive and evidence-based.
""")

# -----------------------------------------------------------------------------
# Sidebar Controls (Filters)
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Filter Settings")

# Date range selection
today = datetime.date.today()
start_date = st.sidebar.date_input("Start Date", today - datetime.timedelta(days=30))
end_date = st.sidebar.date_input("End Date", today)

# Magnitude filter
min_magnitude = st.sidebar.slider("Minimum Magnitude ($M_w$)", 0.0, 9.0, 2.5, 0.5)

# Predefined Region Coordinates [lat, lon, radius_in_km]
regions = {
    "Global (All)": None,
    "Japan & Surroundings": [36.2048, 138.2529, 1000],
    "South Korea & Surroundings": [35.9078, 127.7669, 500],
    "West Coast US (California)": [36.7783, -119.4179, 800],
    "Custom Location": "custom"
}

selected_region = st.sidebar.selectbox("Select Target Region", list(regions.keys()))

lat, lon, radius = None, None, None
if selected_region == "Custom Location":
    lat = st.sidebar.number_input("Latitude", -90.0, 90.0, 37.5665)  # Default: Seoul
    lon = st.sidebar.number_input("Longitude", -180.0, 180.0, 126.9780)
    radius = st.sidebar.slider("Search Radius (km)", 10, 2000, 500)
elif regions[selected_region] is not None:
    lat, lon, radius = regions[selected_region]

# Auto-refresh trigger
if st.sidebar.button("🔄 Refresh Data Now"):
    st.rerun()

# -----------------------------------------------------------------------------
# Data Fetching Function
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)  # Cache data for 5 minutes to stay responsive
def fetch_earthquake_data(start, end, min_mag, lat=None, lon=None, max_rad_km=None):
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%d"),
        "endtime": end.strftime("%Y-%m-%d"),
        "minmagnitude": min_mag,
        "orderby": "time"
    }
    
    if lat is not None and lon is not None and max_rad_km is not None:
        params["latitude"] = lat
        params["longitude"] = lon
        params["maxradiuskm"] = max_rad_km

    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            features = data.get("features", [])
            
            # Parse into a clean list of dictionaries
            eq_list = []
            for f in features:
                props = f["properties"]
                geom = f["geometry"]
                eq_list.append({
                    "time": pd.to_datetime(props["time"], unit="ms"),
                    "magnitude": props["mag"],
                    "place": props["place"],
                    "latitude": geom["coordinates"][1],
                    "longitude": geom["coordinates"][0],
                    "depth_km": geom["coordinates"][2]
                })
            return pd.DataFrame(eq_list)
        else:
            st.error(f"API Error: Received status code {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to connect to USGS API: {e}")
        return pd.DataFrame()

# Fetch data
df = fetch_earthquake_data(start_date, end_date, min_magnitude, lat, lon, radius)

# -----------------------------------------------------------------------------
# Main Dashboard UI
# -----------------------------------------------------------------------------
if df.empty:
    st.warning("⚠️ No earthquake data found matching these parameters. Try expanding your filters.")
else:
    # 1. Key Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Count", f"{len(df)} events")
    with col2:
        st.metric("Max Magnitude", f"M {df['magnitude'].max():.1f}")
    with col3:
        st.metric("Avg Magnitude", f"M {df['magnitude'].mean():.1f}")
    with col4:
        st.metric("Deepest Quake", f"{df['depth_km'].max():.1f} km")

    st.markdown("---")

    # 2. Interactive Map and Data View
    st.subheader("🗺️ Geographic Distribution & Live Map")
    
    # Scale coordinates/points based on magnitude for better visual hierarchy
    map_df = df.copy()
    # Normalize size for display mapping
    map_df['display_size'] = (map_df['magnitude'] ** 2) * 2 
    
    st.map(map_df, latitude='latitude', longitude='longitude', size='display_size')

    # 3. Analytics Charts
    st.markdown("---")
    st.subheader("📊 Statistical Analytics")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**Magnitude Distribution Count**")
        fig_hist = px.histogram(df, x="magnitude", nbins=20, labels={"magnitude": "Magnitude"}, 
                                color_discrete_sequence=['#ef553b'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with chart_col2:
        st.markdown("**Earthquake Timeline & Timeline Trends**")
        fig_scatter = px.scatter(df, x="time", y="magnitude", size="magnitude", color="depth_km",
                                 labels={"time": "Date/Time", "magnitude": "Magnitude", "depth_km": "Depth (km)"},
                                 color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 4. Educational Earthquake Prediction / Risk Assessment Module
    st.markdown("---")
    st.subheader("🔮 Educational Predictive Analysis (Statistical Risk Assessment)")
    
    with st.expander("💡 Read this first: How do scientists 'predict' earthquakes?", expanded=True):
        st.info("""
        **Important Science Note for Students:** Currently, scientists **cannot** predict the exact time, date, and 
        location of a specific future earthquake. Instead, seismologists look at historical trends to calculate 
        **probabilities** and **hazard risk levels**. 
        
        The module below calculates an automated **Poisson Probability** estimation based on the active timeframe 
        selected in your filters.
        """)
    
    # Simple statistical calculation based on frequency
    days_range = (end_date - start_date).days
    if days_range <= 0:
        days_range = 1
        
    # Calculate lambda (average rate of earthquakes per day with specified magnitude or higher)
    lambda_rate = len(df) / days_range
    
    pred_col1, pred_col2 = st.columns(2)
    
    with pred_col1:
        st.markdown("### 📊 Regional Activity Profile")
        st.write(f"• **Current Average Rate:** `{lambda_rate:.3f}` earthquakes per day in this selection criteria.")
        
        # Risk levels based on rate
        if lambda_rate == 0:
            risk_level = "🟢 Very Low"
            action_plan = "Keep standard safety awareness guidelines handy."
        elif lambda_rate < 0.2:
            risk_level = "🟡 Moderate"
            action_plan = "Review basic safety plans and confirm emergency kits are stocked."
        elif lambda_rate < 1.0:
            risk_level = "🟠 High"
            action_plan = "Ensure drop-cover-hold procedures are practiced routinely by students."
        else:
            risk_level = "🔴 Exceptionally High Activity"
            action_plan = "Seismically very active window! Highly vital to perform periodic evacuation drills."
            
        st.markdown(f"• **Calculated Regional Activity Tier:** **{risk_level}**")
        st.markdown(f"• **Drill Action Suggestion:** *{action_plan}*")

    with pred_col2:
        st.markdown("### 🎲 Statistical Probability Forecast")
        st.write("Using a Poisson distribution model, we can estimate the probability of at least one event occurring in the near future:")
        
        # Calculate P(X >= 1) = 1 - e^(-lambda * t) for next 3 days and next 7 days
        prob_3day = (1 - np.exp(-lambda_rate * 3)) * 100
        prob_7day = (1 - np.exp(-lambda_rate * 7)) * 100
        
        st.metric(label=f"Probability of M {min_magnitude}+ Quake in next 3 days", value=f"{prob_3day:.1f}%")
        st.metric(label=f"Probability of M {min_magnitude}+ Quake in next 7 days", value=f"{prob_7day:.1f}%")

    # 5. Raw Data View Table
    st.markdown("---")
    st.subheader("📋 Detailed Event Logs")
    st.dataframe(df[["time", "magnitude", "place", "depth_km", "latitude", "longitude"]].head(100), use_container_width=True)
