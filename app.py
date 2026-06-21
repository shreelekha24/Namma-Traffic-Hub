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
    
    # Initialize session state for location
    if "marker_lat" not in st.session_state:
        st.session_state.marker_lat = 12.9716
        st.session_state.marker_lon = 77.5946
    if "last_query" not in st.session_state:
        st.session_state.last_query = "Bengaluru"
        
    # 1. Text Search to center the map
    address_query = st.sidebar.text_input("🔍 Search Landmark (Press Enter to center map)", st.session_state.last_query)
    
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
    description = st.sidebar.text_area("Enter on-ground updates...", "Severe accident, truck overturned blocking both lanes.")

    st.sidebar.subheader("Diversion Settings")
    traffic_direction = st.sidebar.selectbox("Impacted Traffic Flow", [
        "North ➡️ South",
        "South ➡️ North",
        "East ➡️ West",
        "West ➡️ East"
    ])

    st.sidebar.subheader("Incident Details")
    event_cause = st.sidebar.selectbox("Event Cause", options['event_cause'], index=0)
    veh_type = st.sidebar.selectbox("Vehicle Type", options['veh_type'], index=0)
    priority = st.sidebar.selectbox("Priority Level", options['priority'], index=0)
    
    with st.sidebar.expander("⚙️ Optional Advanced Predictors"):
        st.caption("*(Only needed when the event involves a vehicle breakdown or cargo)*")
        reason_breakdown = st.text_input("Reason for Breakdown", "")
        cargo_material = st.text_input("Cargo Material", "")
        comment = st.text_input("Additional Comments", "")
    
    force_road_closure = st.sidebar.checkbox("⚠️ Force Full Road Closure (Manual Override)")



    # --- Process Inputs on Button Click ---
    if st.sidebar.button("🚨 Predict & Generate Plan", type="primary"):
        with st.spinner("Executing ML Auto-Dispatch & NLP Solver..."):
            
            # Step 1: Auto-Predict Jurisdiction and Corridor
            auto_spatial_df = pd.DataFrame({'latitude': [latitude], 'longitude': [longitude]})
            assigned_police_station = jurisdiction_model.predict(auto_spatial_df)[0]
            assigned_corridor = corridor_model.predict(auto_spatial_df)[0]
            
            st.success(f"🤖 **Auto-Assigned Jurisdiction:** {assigned_police_station} Traffic Police Station")
            st.success(f"🤖 **Auto-Assigned Corridor:** {assigned_corridor}")
            
            # 1. Temporal Features
            hour = time_of_event.hour
            is_rush_hour = 1 if (8 <= hour <= 11) or (17 <= hour <= 20) else 0
            
            # 2. NLP Features (Transformer + PCA)
            try:
                base_desc = description if description else ""
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
                    
            if requires_road_closure:
                if force_road_closure:
                    st.error("🚧 **Auto-Predicted Road Closure:** YES (Forced by Manual Override Checkbox)")
                elif override_triggered:
                    st.error("🚧 **Auto-Predicted Road Closure:** YES (Forced by critical keywords in description)")
                else:
                    st.error("🚧 **Auto-Predicted Road Closure:** YES (Full Blockage)")
            else:
                st.success("✅ **Auto-Predicted Road Closure:** NO (Partial/No Blockage)")
            
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
            
            # 4. Predict
            pred_duration = model.predict(input_df)[0]
            pred_duration = max(1.0, pred_duration) # Prevent negative predictions
            
            if event_type == "planned" and specify_end:
                from datetime import datetime, date
                t1 = datetime.combine(date.today(), time_of_event)
                t2 = datetime.combine(date.today(), planned_end_time)
                diff = (t2 - t1).total_seconds() / 60.0
                if diff < 0:
                    diff += 24 * 60
                # Use the manual planned duration if it's longer than what the ML model predicts
                pred_duration = max(pred_duration, diff)
            
            # 5. Generate Recommendations using PuLP
            recs = generate_recommendations(pred_duration, event_type, event_cause, requires_road_closure, crowd_size=crowd_size, desc_text=desc_text)
            
            # --- Display Results ---
            st.divider()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("⏱️ Forecasting Engine")
                st.metric(label="Predicted Clearance Time", value=f"{int(pred_duration)} mins")
                
                if recs['priority_alert'] == "High":
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
                    
            with col2:
                st.subheader("🗺️ Spatial-Graph Diversion Engine")
                if recs['diversion_advised']:
                    st.info("Calculating optimal diversion using OpenStreetMap Graph...")
                    
                    try:
                        G = get_city_graph()
                        
                        # True Graph Routing: Block the impacted area
                        G_copy = G.copy()
                        
                        # 1. Identify the impacted physical road segment (edge u -> v)
                        try:
                            nearest_edge = ox.distance.nearest_edges(G_copy, longitude, latitude)
                            u, v, key = nearest_edge
                            
                            # 2. Block the road by removing it from our mathematical graph
                            G_copy.remove_edge(u, v, key)
                            if G_copy.has_edge(v, u, key):
                                G_copy.remove_edge(v, u, key)
                                
                            # 3. Calculate Detour topologically
                            # Respect the "Impacted Traffic Flow" direction selected in the UI
                            node_u = G.nodes[u]
                            node_v = G.nodes[v]
                            
                            start_node, end_node = u, v
                            
                            if "South ➡️ North" in traffic_direction:
                                if node_u['y'] > node_v['y']: start_node, end_node = v, u
                            elif "North ➡️ South" in traffic_direction:
                                if node_u['y'] < node_v['y']: start_node, end_node = v, u
                            elif "West ➡️ East" in traffic_direction:
                                if node_u['x'] > node_v['x']: start_node, end_node = v, u
                            elif "East ➡️ West" in traffic_direction:
                                if node_u['x'] < node_v['x']: start_node, end_node = v, u
                                
                            route = None
                            try:
                                route = nx.shortest_path(G_copy, start_node, end_node, weight='length')
                            except nx.NetworkXNoPath:
                                # Fallback if strict one-way street constraints block the preferred flow
                                try:
                                    route = nx.shortest_path(G_copy, end_node, start_node, weight='length')
                                except nx.NetworkXNoPath:
                                    pass
                            
                            if route is None or len(route) < 2:
                                st.warning("⚠️ Blocking this road segment completely cuts off the area. No detour possible.")
                                route_map = folium.Map(location=[latitude, longitude], zoom_start=14, tiles="OpenStreetMap")
                            else:
                                # Plot the detour path
                                route_gdf = ox.routing.route_to_gdf(G_copy, route)
                                route_map = route_gdf.explore(
                                    color="red", 
                                    style_kwds={"weight": 5, "opacity": 0.8}, 
                                    tiles="OpenStreetMap",
                                    width="100%", 
                                    height="100%"
                                )
                        except Exception as e:
                            st.error(f"Routing error: Could not process graph edges. Details: {str(e)}")
                            route_map = folium.Map(location=[latitude, longitude], zoom_start=14, tiles="OpenStreetMap")
                        
                        folium.Marker(location=(latitude, longitude), popup="ACCIDENT ZONE", icon=folium.Icon(color="black", icon="info-sign")).add_to(route_map)
                        
                        st_folium(route_map, width=500, height=400, returned_objects=[])
                        st.success("Route mathematically optimized bypassing the incident node.")
                    except nx.NetworkXNoPath:
                        st.error("No valid diversion path found around the incident.")
                    except Exception as e:
                        st.warning(f"Could not render OSM map: {e}")
                else:
                    st.success("No major diversion required for this incident level.")

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
