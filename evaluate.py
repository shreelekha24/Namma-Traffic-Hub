import pandas as pd
import numpy as np
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

def evaluate_models():
    print("Loading data...")
    df = pd.read_csv("data/events.csv")
    
    # 1. Basic Cleaning
    df['duration_mins'] = pd.to_numeric(df['duration_mins'], errors='coerce')
    df = df.dropna(subset=['duration_mins'])
    df = df[(df['duration_mins'] > 0) & (df['duration_mins'] < 1440)]
    
    # NLP Features
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    df['description'] = df['description'].fillna("none")
    print("Extracting embeddings...")
    embeddings = transformer_model.encode(df['description'].tolist())
    pca = PCA(n_components=20, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    nlp_cols = [f'nlp_embed_{i}' for i in range(20)]
    nlp_df = pd.DataFrame(embeddings_pca, columns=nlp_cols, index=df.index)
    df = pd.concat([df, nlp_df], axis=1)

    # 2. Features
    cat_features = ['event_cause', 'veh_type', 'priority']
    for col in cat_features:
        df[col] = df[col].fillna("Unknown").astype(str).astype("category")
        
    num_features = ['hour', 'day_of_week', 'is_rush_hour', 'centrality', 'distance_to_station_km']
    featuresX = df[cat_features + num_features + nlp_cols]
    y = np.log1p(df['duration_mins'])
    
    # Split exactly as in training
    X_train, X_test, y_train, y_test = train_test_split(featuresX, y, test_size=0.2, random_state=42)

    print("Loading models...")
    try:
        model_cb = cb.CatBoostRegressor()
        model_cb.load_model("catboost_duration.cbm")
        
        model_xg = xgb.XGBRegressor()
        model_xg.load_model("xgboost_duration.json")
        
        model_lg = lgb.Booster(model_file="lightgbm_duration.txt")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    print("Predicting...")
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
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_exp, pred_ensemble)
    
    percent_error = np.abs(pred_ensemble - y_true_exp) / y_true_exp
    accuracy_20 = np.mean(percent_error <= 0.20) * 100
    accuracy_30 = np.mean(percent_error <= 0.30) * 100
    
    print("\n" + "="*40)
    print("ENSEMBLE MODEL PERFORMANCE:")
    print(f"Mean Squared Error (MSE): {mse:.2f} mins^2")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mins")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Accuracy (within 20% margin): {accuracy_20:.2f}%")
    print(f"Accuracy (within 30% margin): {accuracy_30:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_models()
