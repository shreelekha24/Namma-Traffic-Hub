import streamlit as st
import pandas as pd
import os
import json
import joblib
import math
from catboost import CatBoostRegressor, CatBoostClassifier
from sentence_transformers import SentenceTransformer
from recommender import generate_recommendations
from datetime import datetime
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
from shapely.geometry import LineString, Point
from shapely.ops import substring
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import re
import urllib.parse

load_dotenv(override=True)

# --- Page Config ---
st.set_page_config(page_title="Namma Traffic Hub", page_icon="🚔", layout="wide")

# --- Load Models and Assets ---
MODEL_DIR = "models"

@st.cache_resource
def load_assets():
    model = CatBoostRegressor()
    model.load_model(os.path.join(MODEL_DIR, "catboost_duration.cbm"))
    
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    pca = joblib.load(os.path.join(MODEL_DIR, "pca_transformer.pkl"))
    
    with open(os.path.join(MODEL_DIR, "ui_options.json"), "r") as f:
        options = json.load(f)

    with open(os.path.join("data", "station_coords.json"), "r") as f:
        station_coords = json.load(f)
        
    jurisdiction_model = joblib.load(os.path.join(MODEL_DIR, "jurisdiction_model.pkl"))
    corridor_model = joblib.load(os.path.join(MODEL_DIR, "corridor_model.pkl"))
    
    rc_model = CatBoostClassifier()
    rc_model.load_model(os.path.join(MODEL_DIR, "road_closure_model.cbm"))
        
    return model, transformer_model, pca, options, station_coords, jurisdiction_model, corridor_model, rc_model

@st.cache_data
def get_city_graph():
    # Cache the graph so it doesn't download every time
    center_point = (12.987, 77.596)
    try:
        G = ox.load_graphml(os.path.join(MODEL_DIR, "bengaluru_graph.graphml"))
    except:
        G = ox.graph_from_point(center_point, dist=2000, network_type='drive')
    return G

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def inject_virtual_node(G, lat, lon, node_id):
    """Injects a mathematical node into the OSMnx graph at the exact Lat/Lon by splitting the nearest edge."""
    # Find nearest edge
    u, v, key = ox.distance.nearest_edges(G, lon, lat)
    
    # Add new node
    G.add_node(node_id, y=lat, x=lon)
    
    # Calculate distances to u and v in meters
    dist_u = haversine(lat, lon, G.nodes[u]['y'], G.nodes[u]['x']) * 1000
    dist_v = haversine(lat, lon, G.nodes[v]['y'], G.nodes[v]['x']) * 1000
    
    # Get original edge geometry to maintain curvature
    edge_data = G.get_edge_data(u, v, key)
    if 'geometry' in edge_data:
        geom = edge_data['geometry']
    else:
        geom = LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
        
    # Project point onto the LineString to split it
    point = Point(lon, lat)
    dist_along = geom.project(point)
    
    geom_u_to_v_start = substring(geom, 0, dist_along)
    geom_v_start_to_v = substring(geom, dist_along, geom.length)
    
    # Fallback if substring fails (rare)
    if geom_u_to_v_start.geom_type != 'LineString':
        geom_u_to_v_start = LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (lon, lat)])
    if geom_v_start_to_v.geom_type != 'LineString':
        geom_v_start_to_v = LineString([(lon, lat), (G.nodes[v]['x'], G.nodes[v]['y'])])
    
    # Connect the virtual node to the rest of the graph
    if G.has_edge(u, v):
        G.add_edge(node_id, v, length=dist_v, geometry=geom_v_start_to_v)
        G.add_edge(u, node_id, length=dist_u, geometry=geom_u_to_v_start)
    
    if G.has_edge(v, u):
        G.add_edge(node_id, u, length=dist_u, geometry=geom_u_to_v_start)
        G.add_edge(v, node_id, length=dist_v, geometry=geom_v_start_to_v)
        
    return node_id

try:
    model, transformer_model, pca, options, station_coords, jurisdiction_model, corridor_model, rc_model = load_assets()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Models not found or failed to load. Please run train_model.py first. Error: {e}")

# --- Premium Custom CSS ---
st.markdown("""
<style>
    /* Premium UI enhancements */
    
    /* Refined Input Fields */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 6px !important;
        transition: all 0.2s ease;
    }
    
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] div,
    div[data-baseweb="textarea"] textarea {
        color: #FFFFFF !important;
    }
    
    /* Input Hover States */
    div[data-baseweb="input"] > div:hover,
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="textarea"] > div:hover {
        border-color: rgba(255, 255, 255, 0.3) !important;
        background-color: rgba(255, 255, 255, 0.06) !important;
    }
    
    /* Input Focus States */
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="textarea"] > div:focus-within {
        border-color: #F4C095 !important;
        background-color: rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 0 0 1px #F4C095 !important;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1D7874;
        padding: 15px 20px;
        border-radius: 8px;
        border: 1px solid #679289;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: #F4C095;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        color: #F4C095;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600;
        color: #FFFFFF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Primary Button */
    .stButton>button[kind="primary"] {
        background-color: #EE2E31 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: 1px solid #EE2E31 !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #F4C095 !important;
        color: #071E22 !important;
        box-shadow: 0 0 10px rgba(244, 192, 149, 0.4) !important;
    }
    
    /* Dividers */
    hr {
        border-color: #679289;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚔 Namma Traffic Hub")
st.markdown("##### AI-Powered Congestion Forecasting & Tactical Deployment Engine")
st.markdown("---")

if models_loaded:
    # --- Sidebar: User Inputs ---
    # 1. Essential Information
    event_type = st.sidebar.selectbox("Event Type", options['event_type'], index=options['event_type'].index("unplanned") if "unplanned" in options['event_type'] else 0)
    
    specify_end = False
    crowd_size = 0
    if event_type == "planned":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Planned Event Details")
        crowd_size = st.sidebar.number_input("Expected Crowd Size", min_value=0, value=0, step=100)
        specify_end = st.sidebar.checkbox("Specify Planned End Time")
        if specify_end:
            from datetime import datetime
            if "default_end_time" not in st.session_state:
                st.session_state.default_end_time = datetime.now().time()
            planned_end_time = st.sidebar.time_input("Planned End Time", value=st.session_state.default_end_time)
        st.sidebar.markdown("---")
    st.sidebar.subheader("Location")
    
    # Initialize session state for location and 2-step routing
    if "marker_lat" not in st.session_state:
        st.session_state.marker_lat = 12.9716
        st.session_state.marker_lon = 77.5946
    if "last_query" not in st.session_state:
        st.session_state.last_query = "Bengaluru"
    if "step1_complete" not in st.session_state:
        st.session_state.step1_complete = False
    if "impacted_edge" not in st.session_state:
        st.session_state.impacted_edge = None
    if "remaining_width" not in st.session_state:
        st.session_state.remaining_width = None
    if "max_continuous_gap" not in st.session_state:
        st.session_state.max_continuous_gap = None
    if "step1_results" not in st.session_state:
        st.session_state.step1_results = {}
    if "route_start" not in st.session_state:
        st.session_state.route_start = None
    if "route_end" not in st.session_state:
        st.session_state.route_end = None
        
    # UI Session State Keys for Auto-Fill
    if "event_cause_val" not in st.session_state:
        st.session_state.event_cause_val = options['event_cause'][0]
    if "veh_type_val" not in st.session_state:
        st.session_state.veh_type_val = options['veh_type'][0]
    if "priority_val" not in st.session_state:
        st.session_state.priority_val = options['priority'][0]
    if "incident_radius_val" not in st.session_state:
        st.session_state.incident_radius_val = 5
    if "desc_text_val" not in st.session_state:
        st.session_state.desc_text_val = "Severe accident, truck overturned blocking both lanes."
    if "ai_rerun" not in st.session_state:
        st.session_state.ai_rerun = False
        
    # 1. Text Search to center the map
    address_query = st.sidebar.text_input("🔍 Search Landmark (Press Enter to center map)", st.session_state.last_query)
    
    st.sidebar.markdown("---")
    incident_radius = st.sidebar.slider("Impact Radius (Meters)", min_value=1, max_value=20, value=st.session_state.incident_radius_val)
    
    # Update marker to searched location if search query changed
    if address_query != st.session_state.last_query:
        st.session_state.last_query = address_query
        try:
            if address_query:
                st.session_state.marker_lat, st.session_state.marker_lon = ox.geocode(address_query)
                st.rerun()
        except Exception:
            st.sidebar.caption("⚠️ Could not find address.")
            
    # 2. Interactive Map for location selection
    m = folium.Map(location=[st.session_state.marker_lat, st.session_state.marker_lon], zoom_start=13)
    # Add the marker to the map BEFORE rendering it!
    folium.Marker(
        [st.session_state.marker_lat, st.session_state.marker_lon], 
        icon=folium.Icon(color="red", icon="info-sign"),
        tooltip="Incident Location"
    ).add_to(m)
    
    # Add a transparent circle representing the impact radius
    folium.Circle(
        location=[st.session_state.marker_lat, st.session_state.marker_lon],
        radius=incident_radius,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.4
    ).add_to(m)
    
    # Create Dashboard Layout
    dash_col1, dash_col2 = st.columns([3, 1])
    
    with dash_col1:
        st.markdown("### 🗺️ Incident Location Tracker")
        st.markdown("👉 **Click anywhere on the map to drop a pin and set the exact incident coordinates:**")
        
        # Make the map full width in the main container instead of tiny
        map_data = st_folium(m, height=450, use_container_width=True, key="location_map")
        
    with dash_col2:
        st.markdown("### 📡 System Status")
        st.success("🟢 ML Forecasting: **Online**")
        st.success("🟢 NLP Engine: **Online**")
        st.success("🟢 Graph Router: **Online**")
        st.markdown("---")
        st.markdown("### 📝 Quick Guide")
        st.info("1. Pin location on map.")
        st.info("2. Add dispatch notes in sidebar.")
        st.info("3. Hit **Predict** for tactical plan.")
    
    # 3. Extract coordinates from map click
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]
        
        # If the user clicked a new spot, update session state and rerun to move the red marker!
        if round(clicked_lat, 4) != round(st.session_state.marker_lat, 4) or round(clicked_lon, 4) != round(st.session_state.marker_lon, 4):
            st.session_state.marker_lat = clicked_lat
            st.session_state.marker_lon = clicked_lon
            st.rerun()
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        latitude = st.number_input("Lat", value=float(st.session_state.marker_lat), format="%.4f")
    with col2:
        longitude = st.number_input("Lon", value=float(st.session_state.marker_lon), format="%.4f")
    
    st.sidebar.subheader("Timing")
    if "default_time" not in st.session_state:
        st.session_state.default_time = datetime.now().time()
    time_of_event = st.sidebar.time_input("Time of Event", st.session_state.default_time)
    day_of_week = st.sidebar.selectbox("Day of Week", [0,1,2,3,4,5,6], format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
    
    st.sidebar.subheader("Dispatch Notes")
    description = st.sidebar.text_area("Enter on-ground updates...", value=st.session_state.desc_text_val)
    uploaded_images = st.sidebar.file_uploader("Upload Incident Images (Max 10)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if "processed_images" not in st.session_state:
        st.session_state.processed_images = []
        
    current_image_names = [img.name for img in uploaded_images] if uploaded_images else []
    
    if current_image_names and current_image_names != st.session_state.processed_images:
        with st.sidebar.status("👁️ Vision AI Analyzing...", expanded=True) as status:
            st.write("Extracting details from images...")
            try:
                genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                vision_model = genai.GenerativeModel('gemini-flash-latest')
                pil_images = [Image.open(img) for img in uploaded_images]
                
                prompt = """Analyze these traffic incident images. 
1. Generate a clear, professional dispatch note describing the scene. You MUST summarize this into a crisp, concise paragraph of exactly 2 to 3 lines maximum. Include the vehicle count and note any visible hazards (e.g., "Due to water logging, 2 bikes and 1 car are stuck blocking the road").
2. Estimate the total impact radius of the blockage in meters (assume a standard car is 4.5m long and a standard traffic lane is 3.5m wide). 
3. Classify the incident into the following exact categories:
   - EVENT_CAUSE: [accident, congestion, construction, others, pot_holes, procession, protest, road_conditions, test_demo, tree_fall, vehicle_breakdown, water_logging]
   - VEH_TYPE: [Unknown, auto, bmtc_bus, heavy_vehicle, ksrtc_bus, lcv, others, private_bus, private_car, taxi, truck]
   - PRIORITY: [High, Low]

At the very end of your response, you MUST print exactly:
ESTIMATED_RADIUS: <number>
EVENT_CAUSE: <string>
VEH_TYPE: <string>
PRIORITY: <string>"""

                response = vision_model.generate_content([prompt] + pil_images)
                response_text = response.text
                
                # Extract the auto-detected radius
                match_rad = re.search(r'ESTIMATED_RADIUS:\s*([\d\.]+)', response_text)
                if match_rad:
                    detected_radius = float(match_rad.group(1))
                    incident_radius_ai = int(min(max(detected_radius, 1), 20))
                    st.session_state.incident_radius_val = incident_radius_ai
                    
                # Extract parameters
                match_cause = re.search(r'EVENT_CAUSE:\s*(\w+)', response_text)
                if match_cause and match_cause.group(1) in options['event_cause']:
                    st.session_state.event_cause_val = match_cause.group(1)

                match_veh = re.search(r'VEH_TYPE:\s*(\w+)', response_text)
                if match_veh and match_veh.group(1) in options['veh_type']:
                    st.session_state.veh_type_val = match_veh.group(1)

                match_pri = re.search(r'PRIORITY:\s*(\w+)', response_text)
                if match_pri and match_pri.group(1) in options['priority']:
                    st.session_state.priority_val = match_pri.group(1)

                # Clean the tags from the final description
                response_text = re.sub(r'ESTIMATED_RADIUS:\s*[\d\.]+', '', response_text)
                response_text = re.sub(r'EVENT_CAUSE:\s*\w+', '', response_text)
                response_text = re.sub(r'VEH_TYPE:\s*\w+', '', response_text)
                response_text = re.sub(r'PRIORITY:\s*\w+', '', response_text).strip()

                st.session_state.desc_text_val = response_text
                st.session_state.processed_images = current_image_names
                status.update(label="👁️ Vision AI Auto-Fill Complete!", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                st.warning(f"⚠️ **Vision AI Error:** {e}")
                st.session_state.processed_images = current_image_names

    st.sidebar.subheader("Incident Details")
    event_cause = st.sidebar.selectbox("Event Cause", options['event_cause'], index=options['event_cause'].index(st.session_state.event_cause_val))
    veh_type = st.sidebar.selectbox("Vehicle Type", options['veh_type'], index=options['veh_type'].index(st.session_state.veh_type_val))
    priority = st.sidebar.selectbox("Priority Level", options['priority'], index=options['priority'].index(st.session_state.priority_val))
    
    with st.sidebar.expander("⚙️ Optional Advanced Predictors"):
        st.caption("*(Only needed when the event involves a vehicle breakdown or cargo)*")
        reason_breakdown = st.text_input("Reason for Breakdown", "")
        cargo_material = st.text_input("Cargo Material", "")
        comment = st.text_input("Additional Comments", "")
    
    force_road_closure = st.sidebar.checkbox("⚠️ Force Full Road Closure (Manual Override)")

    # --- Process Inputs on Button Click ---
    if st.sidebar.button("🚨 Predict & Generate Plan", type="primary"):
        if not description.strip() and not uploaded_images:
            st.sidebar.error("Please provide either a text description or upload incident images.")
            st.stop()
        if uploaded_images and len(uploaded_images) > 10:
            st.sidebar.error("Maximum 10 pictures allowed.")
            st.stop()
            
        with st.spinner("Executing ML Auto-Dispatch & NLP Solver..."):
            
            ai_description = ""
            
            # Step 1: Auto-Predict Jurisdiction and Corridor
            auto_spatial_df = pd.DataFrame({'latitude': [latitude], 'longitude': [longitude]})
            assigned_police_station = jurisdiction_model.predict(auto_spatial_df)[0]
            assigned_corridor = corridor_model.predict(auto_spatial_df)[0]
            
            # Save to state later
            
            # 1. Temporal Features
            hour = time_of_event.hour
            is_rush_hour = 1 if (8 <= hour <= 11) or (17 <= hour <= 20) else 0
            
            # 2. NLP Features (Transformer + PCA)
            try:
                base_desc = description if description else ""
                if ai_description:
                    base_desc = f"{base_desc} {ai_description}".strip()
                mega_text = f"{base_desc} {reason_breakdown} {cargo_material} {comment}".strip()
                if not mega_text:
                    mega_text = "none"
                    
                desc_text = mega_text # for the downstream NLP override block
                
                embedding = transformer_model.encode([mega_text])
                embedding_pca = pca.transform(embedding)[0]
            except Exception:
                embedding_pca = np.zeros(20)
                desc_text = description if description else "none"
                
            # Step 1.5: Auto-Predict Road Closure based on text + categories
            rc_input = {
                'event_type': event_type,
                'event_cause': event_cause,
                'veh_type': veh_type,
                'priority': priority
            }
            for i in range(20):
                rc_input[f'nlp_embed_{i}'] = embedding_pca[i]
            rc_df = pd.DataFrame([rc_input])
            requires_road_closure_str = rc_model.predict(rc_df)[0]
            # Convert string to bool or keep as string depending on how it was trained
            requires_road_closure = (str(requires_road_closure_str).lower() == "true")
            
            # --- DETERMINISTIC NLP OVERRIDE ---
            # Force road closure ONLY if description explicitly mentions full blockage or critical live wires
            desc_lower = desc_text.lower()
            
            override_triggered = False
            
            # Condition 1: Mentions of the entire road/lanes being blocked
            if ("whole road" in desc_lower or "full road" in desc_lower or "both lanes" in desc_lower or "entire road" in desc_lower or "whole lane" in desc_lower):
                if "block" in desc_lower or "clos" in desc_lower:
                    override_triggered = True
                    
            # Condition 2: Explicit "completely blocked"
            if "completely blocked" in desc_lower:
                override_triggered = True
                
            # Condition 3: Live wires falling
            if ("live wire" in desc_lower or "electric wire" in desc_lower) and "fall" in desc_lower:
                override_triggered = True

            # Manual UI Override
            if force_road_closure:
                requires_road_closure = True
            elif override_triggered:
                requires_road_closure = True
                    
            # 3. Graph Features & Capacity
            G = get_city_graph()
            try:
                nearest_node = ox.distance.nearest_nodes(G, longitude, latitude)
                centrality = G.nodes[nearest_node].get('centrality', 0.0)
                
                nearest_edge = ox.distance.nearest_edges(G, longitude, latitude)
                u, v, key = nearest_edge
                edge_data = G.get_edge_data(u, v, key)
                highway_type = edge_data.get('highway', 'unclassified')
                if isinstance(highway_type, list): highway_type = highway_type[0]
                
                width_map = {
                    'trunk': 12, 'primary': 10, 'secondary': 8,
                    'tertiary': 6, 'residential': 4, 'unclassified': 5
                }
                total_road_width = width_map.get(highway_type, 5)
                blockage_diameter = incident_radius * 2
                remaining_width = total_road_width - blockage_diameter
                max_continuous_gap = remaining_width / 2 if remaining_width > 0 else 0
                
                st.session_state.impacted_edge = nearest_edge
            except Exception as e:
                centrality = 0.0
                highway_type = "unknown"
                total_road_width = 5
                blockage_diameter = incident_radius * 2
                max_continuous_gap = 0
                st.session_state.impacted_edge = None

            # --- SYNC PREDICTIVE LOGIC WITH SPATIAL LOGIC ---
            force_spatial_closure = False
            if max_continuous_gap <= 0:
                requires_road_closure = True
                requires_road_closure_str = "Yes"
                force_spatial_closure = True
            elif requires_road_closure:
                # ML/Override forced closure, so we must mathematically eliminate the gap for the routing engine
                max_continuous_gap = 0
                
            st.session_state.max_continuous_gap = max_continuous_gap
            
            if requires_road_closure:
                if force_spatial_closure:
                    closure_status = "🚧 **Auto-Predicted Road Closure:** YES (Forced by Spatial Capacity - 0m Gap)"
                elif force_road_closure:
                    closure_status = "🚧 **Auto-Predicted Road Closure:** YES (Forced by Manual Override Checkbox)"
                elif override_triggered:
                    closure_status = "🚧 **Auto-Predicted Road Closure:** YES (Forced by critical keywords in description)"
                else:
                    closure_status = "🚧 **Auto-Predicted Road Closure:** YES (Full Blockage predicted by model)"
            else:
                closure_status = "✅ **Auto-Predicted Road Closure:** NO (Partial/No Blockage)"

            # 4. Station Distance
            dist_km = 5.0
            if assigned_police_station in station_coords:
                s_lat = station_coords[assigned_police_station]['lat']
                s_lon = station_coords[assigned_police_station]['lon']
                dist_km = haversine(latitude, longitude, s_lat, s_lon)
            
            # 5. Build Input Vector matching training features
            input_dict = {
                'event_type': event_type,
                'event_cause': event_cause,
                'corridor': assigned_corridor,
                'police_station': assigned_police_station,
                'priority': priority,
                'veh_type': veh_type,
                'requires_road_closure': requires_road_closure_str,
                'age_of_truck': "Unknown",
                'hour': hour,
                'day_of_week': day_of_week,
                'is_rush_hour': is_rush_hour,
                'centrality': centrality,
                'distance_to_station_km': dist_km
            }
            
            for i in range(20):
                input_dict[f'nlp_embed_{i}'] = embedding_pca[i]
                
            input_df = pd.DataFrame([input_dict])
            
            # 6. Predict Duration
            pred_duration = model.predict(input_df)[0]
            pred_duration = max(1.0, pred_duration)
            
            if event_type == "planned" and specify_end:
                from datetime import datetime, date
                t1 = datetime.combine(date.today(), time_of_event)
                t2 = datetime.combine(date.today(), planned_end_time)
                diff = (t2 - t1).total_seconds() / 60.0
                if diff < 0:
                    diff += 24 * 60
                pred_duration = max(pred_duration, diff)
            
            # 7. Generate Recommendations using PuLP
            recs = generate_recommendations(pred_duration, event_type, event_cause, requires_road_closure, crowd_size=crowd_size, desc_text=desc_text, corridor=assigned_corridor)

            st.session_state.step1_results = {
                'assigned_police_station': assigned_police_station,
                'assigned_corridor': assigned_corridor,
                'closure_status': closure_status,
                'pred_duration': pred_duration,
                'recs': recs,
                'highway_type': highway_type,
                'total_road_width': total_road_width,
                'blockage_diameter': blockage_diameter,
                'max_continuous_gap': max_continuous_gap,
                'desc_text': desc_text,
                'event_cause': event_cause
            }
            st.session_state.step1_complete = True
            
            if st.session_state.get('ai_rerun', False):
                st.session_state.ai_rerun = False
                st.rerun()

# --- Step 1 UI Rendering (Decoupled from Button) ---
if st.session_state.step1_complete:
    results = st.session_state.step1_results
    
    st.success(f"🤖 **Auto-Assigned Jurisdiction:** {results['assigned_police_station']} Traffic Police Station")
    st.success(f"🤖 **Auto-Assigned Corridor:** {results['assigned_corridor']}")
    if "YES" in results['closure_status']:
        st.error(results['closure_status'])
    else:
        st.success(results['closure_status'])
        
    st.divider()
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⏱️ Forecasting Engine")
        st.metric(label="Predicted Clearance Time", value=f"{int(results['pred_duration'])} mins")
        
        if results['recs']['priority_alert'] == "High":
            st.error("CRITICAL: High Impact Incident")
        elif results['recs']['priority_alert'] == "Medium":
            st.warning("WARNING: Moderate Impact Incident")
        else:
            st.success("INFO: Low Impact Incident")
            
        st.subheader("👮‍♂️ MILP Tactical Deployment")
        m1, m2, m3 = st.columns(3)
        m1.metric("Traffic Cops Required", results['recs']['police_required'])
        m2.metric("Barricades Required", results['recs']['barricades'])
        m3.metric("Tow Trucks Required", results['recs']['tow_trucks'])
        
        st.markdown("### 📋 Action Plan")
        for action in results['recs']['action_plan']:
            st.markdown(f"- {action}")
            
    with col2:
        st.subheader("🛣️ Dynamic Capacity Assessment")
        st.write(f"**Impacted Road Type:** `{results['highway_type']}`")
        st.write(f"**Estimated Total Width:** `{results['total_road_width']}m`")
        st.write(f"**Blockage Diameter:** `{results['blockage_diameter']}m` (Radius x 2)")
        st.write(f"**Max Continuous Gap:** `{max(0, results['max_continuous_gap']):.1f}m` (Assuming center impact)")
        
        if results['max_continuous_gap'] <= 0:
            st.error("🚨 Full Blockage. 0m clearance. Edge legally closed.")
        else:
            st.warning("🚧 Partial Blockage. Route constrained by vehicle width vs gap.")
            
    st.divider()
    st.markdown("### 📡 BTP Communication Control Center")
    st.info("Auto-generated alerts for Public and Logistics broadcast based on the ML prediction and graph topology.")
    
    comm_col1, comm_col2 = st.columns(2)
    
    # Text Generation
    corridor_name = results['assigned_corridor']
    duration_mins = int(results['pred_duration'])
    
    event_cause = results.get('event_cause', '').lower()
    desc_text = results.get('desc_text', '').strip()
    
    # Clean up the cause for the tweet
    if "rally" in desc_text.lower() or "procession" in desc_text.lower():
        clean_cause = "a procession/rally"
        if len(desc_text.split()) <= 6 and desc_text:
            clean_cause = desc_text
    elif event_cause == "accident":
        clean_cause = "an accident"
    elif event_cause == "waterlogging":
        clean_cause = "severe waterlogging"
    elif event_cause == "vehicle breakdown":
        clean_cause = "a vehicle breakdown"
    elif "tree" in desc_text.lower() and "fall" in desc_text.lower():
        clean_cause = "a fallen tree"
    else:
        clean_cause = "an incident"
    
    # Official X (Twitter) Alert
    tweet_text = f"‘Traffic advisory’\nMovement of vehicles is slow at {corridor_name} due to {clean_cause}. Traffic is being diverted. Expected clearance in {duration_mins} mins. Kindly cooperate / take alternate route.\n#BengaluruTraffic #BTP"
    tweet_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(tweet_text)}"
    
    # FM Radio Script
    radio_script = f"Attention Bengaluru commuters. BTP advises a major diversion on {corridor_name} due to {clean_cause}. Please take alternate routes. The area will be cleared in approximately {duration_mins} minutes. Drive safely."
    
    # Logistics API Payload
    logistics_payload = {
        "event_id": f"BTP-{np.random.randint(1000, 9999)}",
        "status": "disrupted",
        "affected_corridor": corridor_name,
        "fleet_delay_estimate": f"+{duration_mins} mins",
        "reroute_required": True,
        "source": "BTP ASTraM API"
    }
    
    with comm_col1:
        st.markdown("**🐦 Official X (Twitter) Alert**")
        st.code(tweet_text, language="text")
        st.markdown(f'<a href="{tweet_url}" target="_blank" style="display: inline-block; padding: 0.5em 1em; color: white; background-color: #1DA1F2; border-radius: 5px; text-decoration: none; font-weight: bold;">Click to Post on X</a>', unsafe_allow_html=True)
        
        st.markdown("<br>**📻 FM Radio Broadcast Script**", unsafe_allow_html=True)
        st.info(radio_script)

    with comm_col2:
        st.markdown("**📦 Logistics Webhook Payload (For Flipkart)**")
        st.code(json.dumps(logistics_payload, indent=2), language="json")

# --- Step 2: Intelligent Routing Engine ---
if st.session_state.step1_complete:
    st.divider()
    st.markdown("## 📍 Step 2: Intelligent Routing Engine")
    
    st.markdown("### 🚑 Routing Mode")
    routing_mode = st.radio("Select Visualization:", ["Civilian Detour (Draw Start/End)", "Emergency Green Corridor (Auto-Route to Incident)"], horizontal=True)
    st.write("") # small gap
    
    route_col1, route_col2 = st.columns([1, 2])
    
    with route_col1:
        if routing_mode == "Civilian Detour (Draw Start/End)":
            st.info("Click the map to set your Start (A) and End (B) coordinates. The detour will calculate automatically.")
            
            # Initialize Step 2 Map Center
            if "step2_map_center" not in st.session_state:
                st.session_state.step2_map_center = [st.session_state.marker_lat, st.session_state.marker_lon]
                
            step2_m = folium.Map(location=st.session_state.step2_map_center, zoom_start=14)
            
            # Add Incident Area
            folium.Marker(
                [st.session_state.marker_lat, st.session_state.marker_lon], 
                icon=folium.Icon(color="red", icon="warning-sign"),
                tooltip="Incident Zone"
            ).add_to(step2_m)
            folium.Circle(
                location=[st.session_state.marker_lat, st.session_state.marker_lon],
                radius=incident_radius,
                color='red', fill=True, fill_opacity=0.4
            ).add_to(step2_m)
            
            # Add selected start/end points
            if st.session_state.route_start:
                folium.Marker(st.session_state.route_start, icon=folium.Icon(color="green", icon="play"), tooltip="Start").add_to(step2_m)
            if st.session_state.route_end:
                folium.Marker(st.session_state.route_end, icon=folium.Icon(color="black", icon="stop"), tooltip="End").add_to(step2_m)
                
            # Render map and catch clicks
            map_data = st_folium(step2_m, width=350, height=350, returned_objects=["last_clicked"])
            
            if map_data and map_data.get("last_clicked"):
                lat = map_data["last_clicked"]["lat"]
                lon = map_data["last_clicked"]["lng"]
                coord = (lat, lon)
                
                if st.session_state.get("last_click_coord") != coord:
                    st.session_state.last_click_coord = coord
                    if not st.session_state.route_start:
                        st.session_state.route_start = [lat, lon]
                        st.rerun()
                    elif not st.session_state.route_end:
                        st.session_state.route_end = [lat, lon]
                        st.rerun()
                    
            if st.button("🔄 Clear Route Points"):
                st.session_state.route_start = None
                st.session_state.route_end = None
                st.session_state.last_click_coord = None
                st.rerun()
                
            vehicle_type = st.selectbox("Vehicle Classification", ["2-Wheeler", "Car", "Heavy Truck"])
            vehicle_width_map = {"2-Wheeler": 1.0, "Car": 2.5, "Heavy Truck": 4.0}
            required_width = vehicle_width_map[vehicle_type]
            
        else:
            st.info("🚑 **Emergency Mode Active.**\n\nNo manual mapping is required. The algorithm will automatically determine the nearest medical facility and route directly to the incident coordinates.")
            
            # We must still define these variables to prevent errors in route_col2 if they try to access them (though route_col2 shouldn't for this mode)
            vehicle_type = "Car"
            required_width = 2.5
        
    with route_col2:
        if routing_mode == "Civilian Detour (Draw Start/End)":
            if not st.session_state.route_start:
                st.warning("⚠️ Waiting for Start Point... (Click Map)")
            elif not st.session_state.route_end:
                st.warning("⚠️ Waiting for End Point... (Click Map)")
            else:
                # Both points acquired. Auto-Calculate.
                start_lat, start_lon = st.session_state.route_start
                end_lat, end_lon = st.session_state.route_end
                
                st.success("Points Acquired! Calculating intelligent detour...")
                with st.spinner(f"Evaluating graph constraints for {vehicle_type}..."):
                    try:
                        G = get_city_graph()
                        G_route = G.copy()
                        
                        # Retrieve Step 1 capacity data
                        u, v, key = st.session_state.impacted_edge
                        max_continuous_gap = st.session_state.max_continuous_gap
                        
                        detour_triggered = False
                        if max_continuous_gap < required_width:
                            # Vehicle is too wide to pass safely. Cut the edge from the graph.
                            G_route.remove_edge(u, v, key)
                            if G_route.has_edge(v, u, key):
                                G_route.remove_edge(v, u, key)
                            detour_triggered = True
                            st.error(f"🚨 **Constraint Failed:** {vehicle_type} requires {required_width}m. Only {max(0, max_continuous_gap):.1f}m gap available. Forcing mathematical detour.")
                        else:
                            st.success(f"✅ **Constraint Passed:** {vehicle_type} requires {required_width}m. {max_continuous_gap:.1f}m gap available. Proceeding directly through impacted zone.")
                        
                        # Inject Virtual Nodes directly into the routing graph
                        start_node = inject_virtual_node(G_route, start_lat, start_lon, "V_START")
                        end_node = inject_virtual_node(G_route, end_lat, end_lon, "V_END")
                        
                        # Calculate Absolute Shortest Topological Path
                        route = None
                        try:
                            route = nx.shortest_path(G_route, start_node, end_node, weight='length')
                        except nx.NetworkXNoPath:
                            pass
                                
                        if route is None or len(route) < 2:
                            st.error("🚫 **Routing Failed:** Blocking this route makes this movement impossible under current vehicle constraints. No alternative mathematical path exists.")
                        else:
                            # Render the Map
                            route_gdf = ox.routing.route_to_gdf(G_route, route)
                            route_map = route_gdf.explore(
                                color="blue", style_kwds={"weight": 5, "opacity": 0.8}, 
                                tiles="OpenStreetMap", width="100%", height="100%"
                            )
                            
                            # Add Interactive Markers
                            folium.Marker(
                                location=[st.session_state.marker_lat, st.session_state.marker_lon], 
                                popup="INCIDENT ZONE", icon=folium.Icon(color="red", icon="warning-sign")
                            ).add_to(route_map)
                            folium.Marker([start_lat, start_lon], tooltip="Origin", icon=folium.Icon(color="green", icon="play")).add_to(route_map)
                            folium.Marker([end_lat, end_lon], tooltip="Destination", icon=folium.Icon(color="black", icon="stop")).add_to(route_map)
                            
                            st_folium(route_map, width=700, height=450, returned_objects=[])
                            
                    except Exception as e:
                        st.error(f"Routing computation failed. Ensure coordinates are valid. Details: {e}")

        elif routing_mode == "Emergency Green Corridor (Auto-Route to Incident)":
            st.success("🚨 Computing direct Green Corridor to the Incident Site...")
            with st.spinner("Finding nearest hospital and securing route..."):
                try:
                    G = get_city_graph()
                    incident_lat = st.session_state.marker_lat
                    incident_lon = st.session_state.marker_lon
                    
                    st.info("🚑 Searching OpenStreetMap for real hospitals within 5km...")
                    hospitals = ox.features_from_point((incident_lat, incident_lon), tags={'amenity': 'hospital'}, dist=5000)
                    if not hospitals.empty:
                        centroids = hospitals.geometry.centroid
                        hospitals['distance'] = centroids.apply(
                            lambda geom: haversine(incident_lat, incident_lon, geom.y, geom.x)
                        )
                        nearest_hospital = hospitals.loc[hospitals['distance'].idxmin()]
                        
                        hosp_geom = nearest_hospital.geometry
                        if hosp_geom.geom_type == 'Polygon' or hosp_geom.geom_type == 'MultiPolygon':
                            hosp_lat = hosp_geom.centroid.y
                            hosp_lon = hosp_geom.centroid.x
                        else:
                            hosp_lat = hosp_geom.y
                            hosp_lon = hosp_geom.x
                            
                        hosp_name = nearest_hospital.get('name', 'Nearest Hospital')
                        if type(hosp_name) is pd.Series:
                            hosp_name = hosp_name.iloc[0]
                        if pd.isna(hosp_name):
                            hosp_name = "Nearest Hospital"
                            
                        # Find hospital node BEFORE injecting string-based virtual node to prevent OSMnx int casting crash
                        hosp_node = ox.distance.nearest_nodes(G, hosp_lon, hosp_lat)
                        # Inject virtual node for precise incident routing
                        incident_node = inject_virtual_node(G, incident_lat, incident_lon, "V_INCIDENT")
                        
                        try:
                            amb_route = nx.shortest_path(G, hosp_node, incident_node, weight='length')
                            amb_gdf = ox.routing.route_to_gdf(G, amb_route)
                            
                            green_map = folium.Map(location=[incident_lat, incident_lon], zoom_start=14)
                            
                            amb_gdf.explore(
                                m=green_map,
                                color="#00FF00", style_kwds={"weight": 7, "dashArray": "10, 10"}, 
                                name="Ambulance Green Corridor"
                            )
                            folium.Marker(
                                [incident_lat, incident_lon], 
                                tooltip="INCIDENT ZONE", icon=folium.Icon(color="red", icon="warning-sign")
                            ).add_to(green_map)
                            folium.Marker(
                                [hosp_lat, hosp_lon], 
                                tooltip=f"🚑 {hosp_name} (Ambulance Dispatch)", 
                                icon=folium.Icon(color="white", icon_color="green", icon="plus", prefix="fa")
                            ).add_to(green_map)
                            
                            st_folium(green_map, width=700, height=450, returned_objects=[])
                            st.success(f"🚑 Green Corridor secured! Routing ambulance from **{hosp_name}** directly to the incident.")
                            
                        except Exception as e:
                            st.warning("Could not calculate a clear path from the hospital to the crash site.")
                    else:
                        st.warning("No OSM hospitals found within a 5km radius.")
                except Exception as e:
                    import traceback
                    st.warning(f"OSMnx Hospital Query Failed: {e}")
                    st.code(traceback.format_exc(), language="python")

# --- Global Static Footer ---
st.markdown('''
<style>
.static-footer {
    width: 100%;
    border-top: 1px solid rgba(103, 146, 137, 0.3);
    padding: 25px 0 10px 0;
    margin-top: 50px;
    text-align: center;
}
</style>

<div class="static-footer">
    <div style="display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 6px;">
        <span style="color: #679289; font-size: 0.95rem;">Built with 🩵 by</span>
        <span style="color: #F4C095; font-weight: bold; font-size: 1.05rem;">404-logic-found</span>
    </div>
    <div style="display: flex; align-items: center; justify-content: center; gap: 32px;">
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="color: #FFFFFF; font-size: 0.9rem; font-weight: 500;">Shreelekha Adhikary</span>
            <a href="https://www.linkedin.com/in/shreelekha-adhikary-b4272128a" target="_blank" style="text-decoration: none; transition: transform 0.2s;" onmouseover="this.style.transform='scale(1.1)'" onmouseout="this.style.transform='scale(1)'">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="14" height="14" alt="LinkedIn" style="vertical-align: middle;">
            </a>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="color: #FFFFFF; font-size: 0.9rem; font-weight: 500;">Nilanjan De</span>
            <a href="https://www.linkedin.com/in/nilanjan-de-41651a27b" target="_blank" style="text-decoration: none; transition: transform 0.2s;" onmouseover="this.style.transform='scale(1.1)'" onmouseout="this.style.transform='scale(1)'">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="14" height="14" alt="LinkedIn" style="vertical-align: middle;">
            </a>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)
