# 🚦 Namma Traffic Hub

**An AI-Powered Dynamic Congestion Management & Resource Allocation Engine**  
*Built for the Modern City.*

---

## 📖 Overview

Current city traffic management relies on static maps, heuristics, and manual estimations. When an accident or blockage occurs, GPS systems often route vehicles into secondary bottlenecks, and emergency resources (police, tow trucks) are dispatched based on guesswork, leading to delayed response times and resource depletion.

**Namma Traffic Hub** replaces guesswork with mathematics and artificial intelligence. We built a fully autonomous mitigation engine consisting of a **4-Tier Artificial Intelligence Architecture** to automatically parse visual evidence, predict incident clearance times, dispatch resources optimally, and calculate the absolute shortest valid topological detour.

---

## ✨ Key Innovations & Architecture

### 1. Vision AI Automated Dispatch (Ghost Typer Integration)
Instead of forcing operators to manually enter incident parameters, our system allows operators to simply upload photos from the scene. We integrated Google's **Gemini Flash Vision AI** to immediately parse the uploaded images. The AI acts like a "Ghost Typer", automatically extracting key details (event cause, vehicle type, priority level), estimating the incident impact radius in meters, and generating a crisp dispatch note. The entire UI automatically snaps to these predicted values within seconds of the image upload, dramatically accelerating emergency response dispatch.

### 2. Topological Spatial Routing (Dynamic Graph Detours)
We loaded the entire city street network as a mathematical graph (nodes and edges) using **OSMnx** and **NetworkX**. When an incident triggers a road closure (either predicted by ML or manually forced), our algorithm physically injects a virtual node exactly at the coordinates and dynamically deletes the blocked street segment from the graph in real-time. This forces the system to calculate the absolute shortest valid path *around* the blockage using side streets, completely avoiding "off-map" errors and guaranteeing a valid topological detour.

### 3. ML Forecasting Ensemble (Clearance Prediction)
To accurately predict exactly how many minutes an incident will take to clear, we deployed an advanced Ensemble architecture utilizing state-of-the-art tree-based algorithms (like **CatBoost**). Our model dynamically distinguishes between planned events (VIP movements, construction) and unplanned events (spontaneous accidents) to adjust clearance predictions with mathematical precision. 

### 4. NLP Semantic Embedding Engine (Deterministic Overrides)
Operators input unstructured text (e.g., *"The road is completely blocked by falling live electric wires"*). We implemented a deep-learning `SentenceTransformer` that converts English text into a mathematical semantic vector, compressed via PCA. If the system detects catastrophic hazards in the text description or the AI-generated dispatch note, it overrides standard ML predictions and forcefully triggers emergency road closure protocols.

### 5. MILP Resource Optimizer (Mathematical Dispatch)
Using Mixed-Integer Linear Programming (`PuLP`), the system automatically classifies incidents and maps them to their exact legal jurisdiction boundaries. It calculates Haversine spatial distances to nearby police stations and mathematically guarantees the fastest emergency response times while ensuring critical stations retain baseline reserve resources.

---

## ⚙️ Tech Stack

* **Frontend & Interactive UI:** Streamlit, Folium
* **Vision AI & Multimodal Analysis:** Google Gemini API (`gemini-flash-latest`)
* **Routing Algorithms & Spatial Data:** OSMnx, NetworkX, Shapely
* **Machine Learning Models:** CatBoost, XGBoost, LightGBM, Scikit-Learn
* **Natural Language Processing:** SentenceTransformers, PCA
* **Mathematical Optimization:** PuLP (Mixed-Integer Linear Programming)
* **Data Processing & Analytics:** Pandas, NumPy, Geopy

---

## 🚀 How to Run Locally

### Prerequisites
* Python 3.9+
* A valid Google Gemini API Key

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shreelekha24/Namma-Traffic-Hub.git
   cd Namma-Traffic-Hub
   ```

2. **Set up your environment:**
   Create a `.env` file in the root directory and add your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Install Dependencies:**
   Ensure you have Python installed, then install all required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Dashboard:**
   Start the interactive Streamlit server:
   ```bash
   streamlit run app.py
   ```
   *The interactive dashboard will automatically open in your default web browser at `http://localhost:8501`.*

---

## 📊 Model Training & Evaluation Pipeline

The system relies on pre-trained models stored in the `/models` directory. If you wish to retrain the models on fresh data or test the accuracy of the predictive models on the dataset:

1. **Train Models:**
   ```bash
   python train_model.py
   ```
   *This will run the feature engineering pipeline, train the CatBoost, Jurisdiction, Corridor, and NLP models, and save them as `.cbm` and `.pkl` files in the `/models` directory.*

2. **Evaluate Models:**
   ```bash
   python evaluate.py
   ```
   *This will execute the master evaluation pipeline against the test dataset to print the MAE report card.*

---

## 💡 Important Usage Notes

1. **Map Interaction:** To change the location of an incident, simply click anywhere on the interactive map. The UI will automatically capture the new Latitude/Longitude and re-run the prediction block.
2. **Vision AI Auto-Fill:** Upload up to 10 images of the traffic incident into the sidebar. Do not click predict immediately—wait 3 seconds for the `👁️ Vision AI Analyzing...` spinner to complete. The parameters will automatically populate on screen for your review.
3. **Advanced Predictors:** Only expand and fill the optional "⚙️ Optional Advanced Predictors" (like Cargo Material) if the `Vehicle Type` is set to a heavy vehicle or breakdown. The model inherently handles empty inputs gracefully.

---

**Team:** 404-logic-found  
*Replacing guesswork with mathematics to save time, resources, and lives.*
