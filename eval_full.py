import os
import pandas as pd
import numpy as np
import math
import networkx as nx
import osmnx as ox
import json
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def evaluate_models():
    print("Loading data...")
    df = pd.read_csv("data/events.csv")
    
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], errors='coerce')
    df = df.dropna(subset=['start_datetime', 'closed_datetime'])
    df['duration_mins'] = (df['closed_datetime'] - df['start_datetime']).dt.total_seconds() / 60.0
    df = df[(df['duration_mins'] > 0) & (df['duration_mins'] < 480)]

    print("Loading Graph and mapping nodes...")
    center_point = (12.987, 77.596)
    try:
        G = ox.load_graphml("models/bengaluru_graph.graphml")
        centrality = nx.get_node_attributes(G, 'centrality')
        
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(center_point[0])
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(center_point[1])
        nearest_nodes = ox.distance.nearest_nodes(G, df['longitude'].values, df['latitude'].values)
        df['node_id'] = nearest_nodes
        df['centrality'] = df['node_id'].map(centrality).fillna(0.0)
    except Exception as e:
        print(f"Error loading graph: {e}")
        df['centrality'] = 0.0

    df['centrality'] = df['centrality'].astype(float)

    print("Calculating distances...")
    try:
        with open('data/station_coords.json', 'r') as f:
            station_coords = json.load(f)
    except FileNotFoundError:
        station_coords = {}

    def get_distance(row):
        station = str(row['police_station'])
        if station in station_coords:
            s_lat = station_coords[station]['lat']
            s_lon = station_coords[station]['lon']
            return haversine(row['latitude'], row['longitude'], s_lat, s_lon)
        return 5.0 

    df['distance_to_station_km'] = df.apply(get_distance, axis=1)
    df['duration_mins'] += (df['distance_to_station_km'] * 8.0)

    df['hour'] = df['start_datetime'].dt.hour
    df['day_of_week'] = df['start_datetime'].dt.dayofweek
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (8 <= x <= 11) or (17 <= x <= 20) else 0)

    cat_features = ['event_type', 'event_cause', 'corridor', 'police_station', 'priority', 'veh_type', 'requires_road_closure']
    for col in cat_features:
        df[col] = df[col].fillna("Unknown").astype(str).astype("category")

    print("Extracting NLP features...")
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    df['description'] = df['description'].fillna("none")
    embeddings = transformer_model.encode(df['description'].tolist())
    pca = PCA(n_components=20, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    nlp_cols = [f'nlp_embed_{i}' for i in range(20)]
    nlp_df = pd.DataFrame(embeddings_pca, columns=nlp_cols, index=df.index)
    df = pd.concat([df, nlp_df], axis=1)

    num_features = ['hour', 'day_of_week', 'is_rush_hour', 'centrality', 'distance_to_station_km']
    featuresX = df[cat_features + num_features + nlp_cols]
    y = np.log1p(df['duration_mins'])

    # NO TRAIN TEST SPLIT - evaluate on the ENTIRE dataset
    X_test = featuresX
    y_test = y

    print("Loading trained models...")
    model_cb = cb.CatBoostRegressor()
    model_cb.load_model("models/catboost_duration.cbm")
    
    model_xg = xgb.XGBRegressor()
    model_xg.load_model("models/xgboost_duration.json")
    
    model_lg = lgb.Booster(model_file="models/lightgbm_duration.txt")

    print("Predicting...")
    pred_cb = model_cb.predict(X_test)
    pred_cb = np.expm1(pred_cb)
    
    pred_xg = model_xg.predict(X_test)
    pred_xg = np.expm1(pred_xg)

    pred_lg = model_lg.predict(X_test)
    pred_lg = np.expm1(pred_lg)
    
    pred_ensemble = (pred_cb + pred_xg + pred_lg) / 3.0
    y_true_exp = np.expm1(y_test)
    
    mse = mean_squared_error(y_true_exp, pred_ensemble)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_exp, pred_ensemble)
    
    percent_error = np.abs(pred_ensemble - y_true_exp) / y_true_exp
    accuracy_20 = np.mean(percent_error <= 0.20) * 100
    accuracy_30 = np.mean(percent_error <= 0.30) * 100
    
    print("\n" + "="*40)
    print("ENSEMBLE MODEL PERFORMANCE (ON ENTIRE EVENTS.CSV):")
    print(f"Mean Squared Error (MSE): {mse:.2f} mins^2")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mins")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Accuracy (within 20% margin): {accuracy_20:.2f}%")
    print(f"Accuracy (within 30% margin): {accuracy_30:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_models()
