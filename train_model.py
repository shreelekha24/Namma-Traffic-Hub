import os
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
from catboost import CatBoostClassifier
import osmnx as ox
import networkx as nx
import joblib
import json

# Define directories
DATA_PATH = "data/events.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_PATH)

print(f"Original shape: {df.shape}")

# 1. Clean Dates and Calculate Target Variable (Duration)
df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], errors='coerce')

# Drop rows where we can't calculate duration
df = df.dropna(subset=['start_datetime', 'closed_datetime'])

# Calculate duration in minutes
df['duration_mins'] = (df['closed_datetime'] - df['start_datetime']).dt.total_seconds() / 60.0

# Filter out bad data (negative duration or extreme outliers > 24 hours)
df = df[(df['duration_mins'] > 0) & (df['duration_mins'] < 1440)]
print(f"Shape after filtering bad durations: {df.shape}")

print("Downloading OSM Graph for central Bengaluru (5km radius)...")
try:
    center_point = (12.987, 77.596) # Center of the dataset
    G = ox.graph_from_point(center_point, dist=5000, network_type='drive')
    print("Calculating Graph Centrality...")
    centrality = nx.degree_centrality(G)
    nx.set_node_attributes(G, centrality, 'centrality')
    
    # Map events to nearest nodes
    print("Mapping events to graph nodes...")
    # Clean lat/lon
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    # Use center for missing
    df['latitude'] = df['latitude'].fillna(center_point[0])
    df['longitude'] = df['longitude'].fillna(center_point[1])
    
    # Calculate nearest nodes
    nearest_nodes = ox.distance.nearest_nodes(G, df['longitude'].values, df['latitude'].values)
    df['node_id'] = nearest_nodes
    df['centrality'] = df['node_id'].map(centrality).fillna(0.0)
    
    print("Saving graph...")
    ox.save_graphml(G, os.path.join(MODEL_DIR, "bengaluru_graph.graphml"))
except Exception as e:
    print(f"Graph download failed: {e}. Using dummy centrality.")
    df['centrality'] = 0.0

# 1.5 Calculate Haversine Distance to Police Station
print("Calculating spatial distance to assigned police stations...")
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# Load geocoded police stations
try:
    with open('data/station_coords.json', 'r') as f:
        station_coords = json.load(f)
except FileNotFoundError:
    print("Warning: station_coords.json not found. Run geocode_stations.py first. Using default distances.")
    station_coords = {}

def get_distance(row):
    station = str(row['police_station'])
    if station in station_coords:
        s_lat = station_coords[station]['lat']
        s_lon = station_coords[station]['lon']
        return haversine(row['latitude'], row['longitude'], s_lat, s_lon)
    return 5.0 # default 5km if unknown

df['distance_to_station_km'] = df.apply(get_distance, axis=1)

# Create duration category for classification
def get_duration_category(mins):
    if mins < 30:
        return "< 30 mins"
    elif mins < 60:
        return "30-60 mins"
    elif mins < 120:
        return "1-2 hours"
    else:
        return "> 2 hours"

df['duration_category'] = df['duration_mins'].apply(get_duration_category)

# 2. Feature Engineering
# Temporal features
df['hour'] = df['start_datetime'].dt.hour
df['day_of_week'] = df['start_datetime'].dt.dayofweek
df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (8 <= x <= 11) or (17 <= x <= 20) else 0)

# Categorical Features setup
cat_features = ['event_type', 'event_cause', 'corridor', 'police_station', 'priority', 'veh_type', 'requires_road_closure']
for col in cat_features:
    df[col] = df[col].fillna("Unknown").astype(str)

# 3. Native Text Features (CatBoost)
df['description'] = df['description'].fillna("none").astype(str)
text_features = ['description']

# 4. Prepare for Modeling
num_features = ['hour', 'day_of_week', 'is_rush_hour', 'centrality', 'distance_to_station_km']
features = cat_features + num_features + text_features

X = df[features]
y = df['duration_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train CatBoost
print("Training CatBoost Model...")
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    text_features=text_features,
    auto_class_weights='Balanced',
    verbose=50,
    random_seed=42
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=20)

# 6. Evaluation
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"\nModel Performance (V2 Architecture - Classification):")
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, preds))

# 7. Save Models and Assets
print(f"\nSaving models to {MODEL_DIR}/...")
model.save_model(os.path.join(MODEL_DIR, "catboost_duration.cbm"))
import joblib

# 6. Train Jurisdiction Auto-Assign Classifier
from sklearn.ensemble import RandomForestClassifier
print("Training Jurisdiction Auto-Assign Model...")
rf_df = df.dropna(subset=['latitude', 'longitude', 'police_station'])
rf_df = rf_df[rf_df['police_station'] != 'No Police Station']
X_rf = rf_df[['latitude', 'longitude']]
y_rf = rf_df['police_station']
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_rf, y_rf)
joblib.dump(rf_model, os.path.join(MODEL_DIR, "jurisdiction_model.pkl"))

# 7. Train Corridor Auto-Assign Classifier
print("Training Corridor Auto-Assign Model...")
corr_df = df.dropna(subset=['latitude', 'longitude', 'corridor'])
X_corr = corr_df[['latitude', 'longitude']]
y_corr = corr_df['corridor']
rf_corridor = RandomForestClassifier(n_estimators=50, random_state=42)
rf_corridor.fit(X_corr, y_corr)
joblib.dump(rf_corridor, os.path.join(MODEL_DIR, "corridor_model.pkl"))

# 8. Train Road Closure Auto-Assign Classifier
print("Training Road Closure Auto-Assign Model...")
from catboost import CatBoostClassifier

# We use categorical and text features to predict road closure
rc_cat_features = ['event_type', 'event_cause', 'veh_type', 'priority']
rc_text_features = ['description']
X_rc = df[rc_cat_features + rc_text_features]
y_rc = df['requires_road_closure']

# Note: requires_road_closure is typically boolean but was cast to str earlier
rc_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=4,
    cat_features=rc_cat_features,
    text_features=rc_text_features,
    verbose=False,
    random_seed=42
)
rc_model.fit(X_rc, y_rc)
rc_model.save_model(os.path.join(MODEL_DIR, "road_closure_model.cbm"))

print("Training complete! All secondary Assets saved.")

# We don't need to save the Transformer model because it auto-downloads/caches via SentenceTransformers.
# But we need to save UI options for the dashboard.

ui_options = {
    "event_type": sorted(df['event_type'].unique().tolist()),
    "event_cause": sorted(df['event_cause'].unique().tolist()),
    "corridor": sorted(df['corridor'].unique().tolist()),
    "police_station": sorted(df['police_station'].unique().tolist()),
    "priority": sorted(df['priority'].unique().tolist()),
    "veh_type": sorted(df['veh_type'].unique().tolist()),
    "requires_road_closure": sorted(df['requires_road_closure'].unique().tolist()),
}

with open(os.path.join(MODEL_DIR, "ui_options.json"), "w") as f:
    json.dump(ui_options, f, indent=4)

print("Training complete! Assets saved.")
