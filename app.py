import streamlit as st
import pandas as pd
import os
import json
import joblib
import math
from catboost import CatBoostRegressor, CatBoostClassifier
from recommender import generate_recommendations
import numpy as np
from datetime import datetime
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium

# --- Page Config ---
st.set_page_config(page_title="Astram Command Center V2", page_icon="🚔", layout="wide")

# --- Load Models and Assets ---
MODEL_DIR = "models"

@st.cache_resource
def load_assets():
    model = CatBoostClassifier()
    model.load_model(os.path.join(MODEL_DIR, "catboost_duration.cbm"))
    
    with open(os.path.join(MODEL_DIR, "ui_options.json"), "r") as f:
        options = json.load(f)

    with open(os.path.join("data", "station_coords.json"), "r") as f:
        station_coords = json.load(f)
        
    jurisdiction_model = joblib.load(os.path.join(MODEL_DIR, "jurisdiction_model.pkl"))
    corridor_model = joblib.load(os.path.join(MODEL_DIR, "corridor_model.pkl"))
    road_closure_model = CatBoostClassifier()
    road_closure_model.load_model(os.path.join(MODEL_DIR, "road_closure_model.cbm"))
        
    return model, options, station_coords, jurisdiction_model, corridor_model, road_closure_model

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

try:
    model, options, station_coords, jurisdiction_model, corridor_model, road_closure_model = load_assets()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Models not found or failed to load. Please run train_model.py first. Error: {e}")

# --- Dark/Light Mode Toggle ---
dark_mode = st.toggle("Dark Mode", value=True)

# --- Professional UI/UX CSS Overrides ---
css_base = """
<style>
    /* Pin the toggle to the top right, fixed */
    div[data-testid="stCheckbox"] {
        position: fixed;
        top: 12px;
        right: 100px;
        z-index: 999999;
        background-color: rgba(0,0,0,0.1);
        padding: 5px 15px;
        border-radius: 20px;
        backdrop-filter: blur(5px);
    }
    
    /* Metric Cards Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        color: #3b82f6;
        text-shadow: 0px 0px 10px rgba(59, 130, 246, 0.4);
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3);
        font-weight: 600;
        letter-spacing: 0.5px;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8 0%, #1e40af 100%);
        border: none;
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.5);
        transform: translateY(-1px);
    }
"""

if dark_mode:
    css_theme = """
    /* Dark Theme Colors */
    .stApp { background-color: #0b0f19; color: #f0f6fc; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    div[data-testid="metric-container"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.5); }
    .streamlit-expanderHeader { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; }
    .streamlit-expanderContent { border: 1px solid #30363d; background-color: #0b0f19; }
    hr { border-color: #30363d; }
    
    /* Force Text Visibility */
    h1, h2, h3, h4, p, label, span, li, .stMarkdown { color: #f0f6fc !important; }
    div[data-testid="stMetricValue"] { color: #3b82f6 !important; }
    
    /* Input Fields styling */
    input, div[data-baseweb="select"] > div, div[data-baseweb="base-input"], div[role="listbox"] {
        background-color: #161b22 !important;
        color: #f0f6fc !important;
        -webkit-text-fill-color: #f0f6fc !important;
    }
    </style>
    """
else:
    css_theme = """
    /* Light Theme Colors */
    .stApp { background-color: #f8fafc; color: #0f172a; }
    [data-testid="stSidebar"] { background-color: #f1f5f9; }
    div[data-testid="metric-container"] { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .streamlit-expanderHeader { background-color: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 6px; }
    .streamlit-expanderContent { border: 1px solid #e2e8f0; background-color: #ffffff; }
    hr { border-color: #e2e8f0; }
    
    /* Force Text Visibility */
    h1, h2, h3, h4, p, label, span, li, .stMarkdown { color: #0f172a !important; }
    div[data-testid="stMetricValue"] { color: #3b82f6 !important; }
    
    /* Input Fields styling */
    input, div[data-baseweb="select"] > div, div[data-baseweb="base-input"], div[role="listbox"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }
    </style>
    """

st.markdown(css_base + css_theme, unsafe_allow_html=True)

st.title("🚔 Astram Command Center")
st.markdown("### AI-Powered Congestion Forecasting & MILP Tactical Resource Deployment")

if models_loaded:
    if 'map_lat' not in st.session_state:
        st.session_state.map_lat = 12.9870
    if 'map_lon' not in st.session_state:
        st.session_state.map_lon = 77.5960

    # --- Sidebar: User Inputs ---
    # 1. Essential Information
    event_type = st.sidebar.selectbox("Event Type", options['event_type'], index=options['event_type'].index("unplanned") if "unplanned" in options['event_type'] else 0)
    
    st.sidebar.subheader("Location")
    address_query = st.sidebar.text_input("Search Landmark/Address", "Cubbon Park, Bengaluru")
    
    # Geocode the address into coordinates
    if st.sidebar.button("Search Landmark"):
        try:
            if address_query:
                auto_lat, auto_lon = ox.geocode(address_query)
                st.session_state.map_lat = float(auto_lat)
                st.session_state.map_lon = float(auto_lon)
                st.rerun()
        except Exception:
            st.sidebar.caption("⚠️ Could not find address.")
        
    col1, col2 = st.sidebar.columns(2)
    with col1:
        latitude = st.number_input("Lat", value=st.session_state.map_lat, format="%.4f")
    with col2:
        longitude = st.number_input("Lon", value=st.session_state.map_lon, format="%.4f")
        
    # Keep session state in sync with manual input changes
    if abs(latitude - st.session_state.map_lat) > 0.0001 or abs(longitude - st.session_state.map_lon) > 0.0001:
        st.session_state.map_lat = latitude
        st.session_state.map_lon = longitude
        st.rerun()

    
    st.sidebar.subheader("Timing")
    if 'event_time_val' not in st.session_state:
        st.session_state.event_time_val = datetime.now().time()
    time_of_event = st.sidebar.time_input("Time of Event", value=st.session_state.event_time_val)
    st.session_state.event_time_val = time_of_event
    
    if 'day_val' not in st.session_state:
        st.session_state.day_val = datetime.now().weekday()
    day_of_week = st.sidebar.selectbox("Day of Week", [0,1,2,3,4,5,6], index=st.session_state.day_val, format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
    st.session_state.day_val = day_of_week
    
    st.sidebar.subheader("Dispatch Notes")
    description = st.sidebar.text_area("Enter on-ground updates...", "Severe accident, truck overturned blocking both lanes.")

    st.sidebar.subheader("Diversion Settings")
    traffic_direction = st.sidebar.selectbox("Impacted Traffic Flow", [
        "North ➡️ South",
        "South ➡️ North",
        "East ➡️ West",
        "West ➡️ East"
    ])

    # 2. Advanced / Optional Fields hidden to reduce clutter
    with st.sidebar.expander("⚙️ Advanced Details (Optional)"):
        event_cause = st.selectbox("Event Cause", options['event_cause'], index=0)
        num_vehicles = st.number_input("Number of Vehicles Involved (For Tow Dispatch)", min_value=0, max_value=20, value=2 if event_cause == "accident" else 0)
        # Corridor is now auto-assigned by ML
        veh_type = st.selectbox("Vehicle Type", options['veh_type'], index=0)
        priority = st.selectbox("Priority Level", options['priority'], index=0)
        # Road Closure is now auto-assigned by ML

    # --- Main Screen Layout ---
    st.divider()
    main_col1, main_col2 = st.columns([1, 1])
    
    with main_col1:
        st.subheader("Interactive Command Map")
        st.markdown("📍 Click anywhere on the map to set the incident location.")
        
        # Base interactive map
        base_m = folium.Map(location=[st.session_state.map_lat, st.session_state.map_lon], zoom_start=13, tiles="cartodbpositron")
        folium.Marker(
            location=[st.session_state.map_lat, st.session_state.map_lon], 
            popup="Target Location", 
            icon=folium.Icon(color="blue", icon_color="#ffffff", icon="crosshairs", prefix='fa')
        ).add_to(base_m)
        
        map_data = st_folium(base_m, width="100%", height=500, key="base_map")
        
        # Check if user clicked the map
        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lon = map_data["last_clicked"]["lng"]
            if abs(lat - st.session_state.map_lat) > 0.0001 or abs(lon - st.session_state.map_lon) > 0.0001:
                st.session_state.map_lat = lat
                st.session_state.map_lon = lon
                st.rerun()
                
    with main_col2:
        # We will show predictions here when button is clicked
        prediction_placeholder = st.empty()

    # --- Process Inputs on Button Click ---
    if st.sidebar.button("🚨 Predict & Generate Plan", type="primary"):
        # --- Form Validation ---
        current_weekday = datetime.now().weekday()
        if event_type == "unplanned":
            if event_cause in ["tree_fall", "water_logging"]:
                allowed_days = [current_weekday, (current_weekday + 1) % 7, (current_weekday + 2) % 7]
                if day_of_week not in allowed_days:
                    st.sidebar.error("⚠️ Invalid Day: Water logging or tree falls can only be reported for today or the next 2 days.")
                    st.stop()
            else:
                if day_of_week != current_weekday:
                    st.sidebar.error("⚠️ Invalid Day: Unplanned events (like accidents) can only be reported on the present day.")
                    st.stop()
                    
        with prediction_placeholder.container():
            st.info("Executing ML Auto-Dispatch & NLP Solver...")
            
            # Step 1: Auto-Predict Jurisdiction
            auto_station_df = pd.DataFrame({'latitude': [latitude], 'longitude': [longitude]})
            assigned_police_station = jurisdiction_model.predict(auto_station_df)[0]
            st.success(f"🤖 **Auto-Assigned Jurisdiction:** {assigned_police_station} Traffic Police Station")
            
            # Step 1.5: Auto-Predict Corridor
            assigned_corridor = corridor_model.predict(auto_station_df)[0]
            st.success(f"🛣️ **Auto-Assigned Corridor:** {assigned_corridor}")
            
            # 1. Temporal Features
            hour = time_of_event.hour
            is_rush_hour = 1 if (8 <= hour <= 11) or (17 <= hour <= 20) else 0
            
            # 2. Text Features (CatBoost Native)
            desc_text = description if description else "none"
            desc_lower = desc_text.lower()
            
            import re
            extracted_num = 0
            # Look for numbers followed by vehicle words
            match = re.search(r'(\d+)\s*(truck|car|vehicle|bike|bus|auto)', desc_lower)
            if match:
                extracted_num = int(match.group(1))
            
            # Use NLP extracted number if found, otherwise fallback to UI input
            final_num_vehicles = extracted_num if extracted_num > 0 else num_vehicles
            
            # Step 1.8: Auto-Predict Road Closure
            is_vip_event_ui = (event_type == 'planned') and (event_cause in ['vip_movement', 'sports_event', 'procession', 'protest', 'public_event'])
            
            rc_dict = {
                'event_type': event_type,
                'event_cause': event_cause,
                'veh_type': veh_type,
                'priority': priority,
                'description': desc_text
            }
            rc_df = pd.DataFrame([rc_dict])
            # road_closure_model outputs 'True' or 'False' (as string)
            assigned_road_closure = road_closure_model.predict(rc_df)[0]
            # Convert to boolean just in case
            assigned_road_closure = True if assigned_road_closure == 'True' else False
            
            if is_vip_event_ui:
                assigned_road_closure = True
                st.error("🚧 **Auto-Assigned Road Closure:** TRUE (Mass Crowd Event Override)")
            elif assigned_road_closure:
                st.error("🚧 **Auto-Assigned Road Closure:** TRUE (ML Prediction: High Risk Event)")
            else:
                st.success("🚧 **Auto-Assigned Road Closure:** FALSE (ML Prediction: Flow maintainable)")

            # 3. Graph Features
            G = get_city_graph()
            try:
                nearest_node = ox.distance.nearest_nodes(G, longitude, latitude)
                centrality = G.nodes[nearest_node].get('centrality', 0.0)
            except Exception:
                centrality = 0.0

            # 4. Station Distance
            dist_km = 5.0
            if assigned_police_station in station_coords:
                s_lat = station_coords[assigned_police_station]['lat']
                s_lon = station_coords[assigned_police_station]['lon']
                dist_km = haversine(latitude, longitude, s_lat, s_lon)
            
            # 3. Build Input Vector matching training features
            input_dict = {
                'event_type': event_type,
                'event_cause': event_cause,
                'corridor': assigned_corridor,
                'police_station': assigned_police_station,
                'priority': priority,
                'veh_type': veh_type,
                'requires_road_closure': str(assigned_road_closure),
                'hour': hour,
                'day_of_week': day_of_week,
                'is_rush_hour': is_rush_hour,
                'centrality': centrality,
                'distance_to_station_km': dist_km,
                'description': desc_text
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # 4. Predict Category
            pred_category_raw = model.predict(input_df)[0]
            pred_category = pred_category_raw[0] if isinstance(pred_category_raw, (list, np.ndarray)) else pred_category_raw
            
            # Determine vehicle size category for Tow Trucks
            veh_cat = extracted_type if 'extracted_type' in locals() and extracted_num > 0 else veh_type.lower()
            
            import math
            if any(w in veh_cat for w in ["truck", "bus", "heavy", "lcv"]):
                tow_trucks_needed = final_num_vehicles * 1.0
            elif any(w in veh_cat for w in ["car", "taxi", "auto", "vehicle"]):
                tow_trucks_needed = final_num_vehicles * 0.5
            elif any(w in veh_cat for w in ["bike", "scoot"]):
                tow_trucks_needed = final_num_vehicles * 0.2
            else:
                tow_trucks_needed = final_num_vehicles * 0.5
                
            tow_trucks_needed = math.ceil(tow_trucks_needed)
            
            # Scale tow trucks based on Time Predicted (urgency)
            if tow_trucks_needed > 0:
                if pred_category == "> 2 hours":
                    tow_trucks_needed += 2  # Max urgency
                elif pred_category == "1-2 hours":
                    tow_trucks_needed += 1  # High urgency
                    
            # 5. Generate Recommendations using PuLP
            recs = generate_recommendations(pred_category, event_type, event_cause, assigned_road_closure, int(tow_trucks_needed))
            
            # --- Display Results ---
            st.divider()
            with st.container():
                st.subheader("⏱️ Forecasting Engine")
                st.metric(label="Predicted Clearance Time", value=f"{pred_category}")
                
                if recs['priority_alert'] == "Critical":
                    st.error("🚨 CRITICAL: VIP / High-Profile Public Event")
                elif recs['priority_alert'] == "High":
                    st.error("CRITICAL: High Impact Incident")
                elif recs['priority_alert'] == "Medium":
                    st.warning("WARNING: Moderate Impact Incident")
                else:
                    st.success("INFO: Low Impact Incident")
                    
                st.subheader("👮‍♂️ MILP Tactical Deployment")
                m1, m2, m3 = st.columns(3)
                m1.metric("Traffic Cops Required", recs['police_required'])
                m2.metric("Barricades Required", recs['barricades'])
                m3.metric("Tow Trucks Required", recs['tow_trucks'])
                
                st.markdown("### 📋 Action Plan")
                for action in recs['action_plan']:
                    st.markdown(f"- {action}")
                    
            with st.container():
                st.subheader("🗺️ Spatial-Graph Diversion Engine")
                if recs['diversion_advised']:
                    st.info("Calculating optimal diversion using OpenStreetMap Graph...")
                    
                    try:
                        G = get_city_graph()
                        
                        # Find the incident node
                        incident_node = ox.distance.nearest_nodes(G, longitude, latitude)
                        
                        # True Graph Routing: Temporarily remove the incident node
                        G_copy = G.copy()
                        if incident_node in G_copy:
                            G_copy.remove_node(incident_node)
                            
                        # Simulate a vehicle trying to travel through this area based on user-selected direction
                        # 0.008 degrees is roughly 800 meters
                        if traffic_direction == "North ➡️ South":
                            orig_node = ox.distance.nearest_nodes(G_copy, longitude, latitude + 0.008)
                            dest_node = ox.distance.nearest_nodes(G_copy, longitude, latitude - 0.008)
                        elif traffic_direction == "South ➡️ North":
                            orig_node = ox.distance.nearest_nodes(G_copy, longitude, latitude - 0.008)
                            dest_node = ox.distance.nearest_nodes(G_copy, longitude, latitude + 0.008)
                        elif traffic_direction == "East ➡️ West":
                            orig_node = ox.distance.nearest_nodes(G_copy, longitude + 0.008, latitude)
                            dest_node = ox.distance.nearest_nodes(G_copy, longitude - 0.008, latitude)
                        else: # West ➡️ East
                            orig_node = ox.distance.nearest_nodes(G_copy, longitude - 0.008, latitude)
                            dest_node = ox.distance.nearest_nodes(G_copy, longitude + 0.008, latitude)
                        
                        # Calculate Shortest Path bypassing the blocked node
                        try:
                            route = nx.shortest_path(G_copy, orig_node, dest_node, weight='length')
                            
                            if len(route) < 2:
                                st.warning("⚠️ Incident location is outside the core city network or isolated. No diversion possible.")
                                route_map = folium.Map(location=[latitude, longitude], zoom_start=14, tiles="cartodbpositron")
                            else:
                                # Plot with Modern OSMnx 2.0+ GeoDataFrames to capture actual road curves
                                route_gdf = ox.routing.route_to_gdf(G_copy, route)
                                route_map = route_gdf.explore(
                                    color="#3b82f6", 
                                    style_kwds={"weight": 6, "opacity": 0.9}, 
                                    tiles="cartodbpositron",
                                    width="100%", 
                                    height="100%"
                                )
                        except nx.NetworkXNoPath:
                            st.warning("⚠️ Blocking this intersection completely cuts off the road. No detour possible.")
                            route_map = folium.Map(location=[latitude, longitude], zoom_start=14, tiles="cartodbpositron")
                        
                        folium.Marker(location=(latitude, longitude), popup="ACCIDENT ZONE", icon=folium.Icon(color="red", icon_color="#ffffff", icon="info-sign")).add_to(route_map)
                        
                        st_folium(route_map, width=500, height=400, returned_objects=[])
                        st.success("Route mathematically optimized bypassing the incident node.")
                    except nx.NetworkXNoPath:
                        st.error("No valid diversion path found around the incident.")
                    except Exception as e:
                        st.warning(f"Could not render OSM map: {e}")
                else:
                    st.success("No major diversion required for this incident level.")
