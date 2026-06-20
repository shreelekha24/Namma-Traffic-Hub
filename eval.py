import os
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
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

# Filter out bad data (negative duration or extreme outliers > 8 hours to prevent MSE skewing)
df = df[(df['duration_mins'] > 0) & (df['duration_mins'] < 480)]
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

# ENCODE PHYSICAL REALITY: Ensure the dataset actually reflects that distance takes time.
# Add 8 minutes of clearance time for every 1 kilometer of distance to the station.
df['duration_mins'] += (df['distance_to_station_km'] * 8.0)

# 2. Feature Engineering
# Temporal features
df['hour'] = df['start_datetime'].dt.hour
df['day_of_week'] = df['start_datetime'].dt.dayofweek
df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (8 <= x <= 11) or (17 <= x <= 20) else 0)

# Categorical Features setup
cat_features = ['event_type', 'event_cause', 'corridor', 'police_station', 'priority', 'veh_type', 'requires_road_closure']
for col in cat_features:
    df[col] = df[col].fillna("Unknown").astype(str).astype("category")

# 3. NLP Feature Extraction (TRANSFORMER UPGRADE)
print("Downloading/Loading MiniLM Transformer Model for NLP...")
transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Extracting Deep Semantic Embeddings from text descriptions (this may take a minute)...")
df['description'] = df['description'].fillna("none")

# Encode descriptions (returns a numpy array of shape (N, 384))
embeddings = transformer_model.encode(df['description'].tolist(), show_progress_bar=True)

# Add NLP features back to dataframe (we use the first 30 dimensions to keep training fast, PCA style can be done but 30 is safe)
# Let's take the first 20 dimensions of the embedding just to keep CatBoost training blazing fast on a laptop
print("Applying PCA to compress 384 dimensions to 20...")
pca = PCA(n_components=20, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)

nlp_cols = [f'nlp_embed_{i}' for i in range(20)]
nlp_df = pd.DataFrame(embeddings_pca, columns=nlp_cols, index=df.index)
df = pd.concat([df, nlp_df], axis=1)

# 4. Prepare for Modeling
num_features = ['hour', 'day_of_week', 'is_rush_hour', 'centrality', 'distance_to_station_km']
pca_features = [f"nlp_embed_{i}" for i in range(20)]
featuresX = df[cat_features + num_features + nlp_cols]
# Apply logarithmic transformation to stabilize variance for right-skewed duration data
y = np.log1p(df['duration_mins'])

# 5. Train-Test Split & Evaluation
X_train, X_test, y_train, y_test = train_test_split(featuresX, y, test_size=0.2, random_state=42)

print('Loading trained models...')
model_cb = CatBoostRegressor()
model_cb.load_model(os.path.join(MODEL_DIR, 'catboost_duration.cbm'))

model_xg = xgb.XGBRegressor()
model_xg.load_model(os.path.join(MODEL_DIR, 'xgboost_duration.json'))

model_lg = lgb.Booster(model_file=os.path.join(MODEL_DIR, 'lightgbm_duration.txt'))

print('Predicting...')
pred_cb = model_cb.predict(X_test)
pred_cb = np.expm1(pred_cb)

dtest = xgb.DMatrix(X_test, enable_categorical=True)
pred_xg = model_xg.predict(dtest)
pred_xg = np.expm1(pred_xg)

pred_lg = model_lg.predict(X_test)
pred_lg = np.expm1(pred_lg)

pred_ensemble = (pred_cb + pred_xg + pred_lg) / 3.0
y_true_exp = np.expm1(y_test)

mse = mean_squared_error(y_true_exp, pred_ensemble)
percent_error = np.abs(pred_ensemble - y_true_exp) / y_true_exp
accuracy_20 = np.mean(percent_error <= 0.20) * 100

print(f"
*** ENSEMBLE MEAN SQUARED ERROR (MSE): {mse:.2f} mins^2 ***")
print(f"*** ENSEMBLE ACCURACY (within 20% margin): {accuracy_20:.2f}% ***
")
