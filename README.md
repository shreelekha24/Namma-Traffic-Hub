# 🚦 Namma Traffic Hub

**An AI-Powered Dynamic Congestion Management & Resource Allocation Engine**  
*Built for the Modern City.*

---

## 📖 Overview

Current city traffic management relies on static maps, heuristics, and manual estimations. When an accident or blockage occurs, GPS systems often route vehicles into secondary bottlenecks, and emergency resources (police, tow trucks) are dispatched based on guesswork, leading to delayed response times and resource depletion.

**Namma Traffic Hub** replaces guesswork with mathematics. We built a fully autonomous mitigation engine consisting of a **3-Tier Artificial Intelligence Architecture** to predict incident clearance times, dispatch resources optimally, and calculate the absolute shortest valid topological detour.

---

## ✨ Key Innovations & Architecture

### 1. Topological Spatial Routing (Dynamic Graph Detours)
We loaded the entire city street network as a mathematical graph (nodes and edges) using **OSMnx** and **NetworkX**. 
When an incident triggers a road closure, our algorithm physically deletes that specific street edge from the graph in real-time. This forces the system to calculate the absolute shortest valid path *around* the blockage using side streets, completely avoiding "off-map" errors and guaranteeing a valid topological detour.

### 2. ML Forecasting Ensemble (Clearance Prediction)
To accurately predict exactly how many minutes an incident will take to clear, we deployed an advanced Ensemble architecture utilizing three state-of-the-art tree-based algorithms: **CatBoost, XGBoost, and LightGBM**. 
Our model dynamically distinguishes between planned events (VIP movements, construction) and unplanned events (spontaneous accidents) to adjust clearance predictions with mathematical precision.
* **Accuracy Achieved:** Final Ensemble MAE of just **34.8 Minutes** on volatile city traffic data.

### 3. NLP Semantic Embedding Engine (Deterministic Overrides)
Operators input unstructured text (e.g., *"The road is completely blocked by falling live electric wires"*). We implemented a deep-learning `SentenceTransformer` that converts English text into a mathematical semantic vector, compressed via PCA. If the system detects catastrophic hazards, it overrides standard ML predictions and forcefully triggers emergency protocols.

### 4. MILP Resource Optimizer (Mathematical Dispatch)
Using Mixed-Integer Linear Programming (`PuLP`), the system automatically classifies incidents and maps them to their exact legal jurisdiction boundaries. It calculates Haversine spatial distances to nearby police stations and mathematically guarantees the fastest emergency response times while ensuring critical stations retain reserve resources.

---

## ⚙️ Tech Stack

* **Frontend/UI:** Streamlit, Folium
* **Routing Algorithms:** OSMnx, NetworkX
* **Machine Learning:** CatBoost, XGBoost, LightGBM, Scikit-Learn
* **Natural Language Processing:** SentenceTransformers, PCA
* **Mathematical Optimization:** PuLP (Mixed-Integer Linear Programming)
* **Data Processing:** Pandas, NumPy, Geopy

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shreelekha24/Namma-Traffic-Hub.git
   cd Namma-Traffic-Hub
   ```

2. **Install Dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard:**
   ```bash
   streamlit run app.py
   ```
   *The interactive dashboard will automatically open in your default web browser.*

---

## 📊 Evaluation Pipeline
If you wish to test the accuracy of the predictive models on the dataset:
```bash
python evaluate.py
```
*This will execute the master evaluation pipeline, replicating feature engineering and testing the ensemble models against the test dataset to print the MAE report card.*

---

**Team:** 404-logic-found  
*Replacing guesswork with mathematics to save time, resources, and lives.*
