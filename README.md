<p align="center">
  <img width="800" alt="Main Dashboard" src="https://github.com/user-attachments/assets/b3e5c87a-7d27-4814-ae50-99f7421d8bef" />
  <br>
  <kbd><b>Main Dashboard Overview</b></kbd>
</p>

<p align="center">
  <img width="800" alt="Interactive Fire Map" src="https://github.com/user-attachments/assets/ad274283-837a-406b-9c16-8e24ecc6e060" />
  <br>
  <kbd><b>Interactive Fire Risk Map</b></kbd>
</p>

<p align="center">
  <img width="800" alt="System Architecture" src="https://github.com/user-attachments/assets/2a32e9ff-f916-48a8-b548-f8e501e1591b" />
  <br>
  <kbd><b>System Architecture</b></kbd>
</p>

# 🔥 Early Detection and Risk Mapping of Forest Fires

[cite_start]A full-stack, AI-driven application designed to provide proactive forest fire management for India by integrating near real-time satellite data with machine learning[cite: 105, 136, 194].

## 🚀 Project Overview
[cite_start]This system addresses the limitations of traditional, reactive fire management[cite: 134, 135]. [cite_start]It leverages satellite thermal data to identify active fires and applies an XGBoost classifier to assess risk levels based on intensity and geospatial coordinates[cite: 106, 107, 138, 139].

### Key Features
* [cite_start]**Live Data Acquisition:** Automatically fetches near real-time fire detections from the NASA FIRMS API[cite: 197, 409].
* [cite_start]**AI-Powered Risk Prediction:** Classifies fire incidents into Low, Moderate, or High-risk categories using a trained XGBoost model[cite: 107, 196, 418].
* [cite_start]**Geospatial Visualization:** Interactive maps featuring Marker Cluster and Heatmap views to visualize fire density and locations across India[cite: 199, 421].
* [cite_start]**Analytical Dashboard:** Real-time charts displaying fire risk distribution and hourly detection trends[cite: 111, 428].
* [cite_start]**Data Export:** Functionality to download filtered datasets in CSV and JSON formats for offline analysis[cite: 430, 431].

## 🛠️ Tech Stack
* [cite_start]**Backend:** FastAPI (Python) [cite: 256, 297]
* [cite_start]**Frontend:** Streamlit [cite: 306, 399]
* [cite_start]**Machine Learning:** XGBoost Classifier [cite: 255, 295]
* [cite_start]**Data Science:** Pandas, NumPy, Scikit-learn [cite: 254, 288, 292]
* [cite_start]**Visualization:** Folium, Plotly [cite: 311, 312, 402]
* [cite_start]**Data Source:** NASA FIRMS API [cite: 106, 314]

## ⚙️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/forest-fire-monitor.git](https://github.com/your-username/forest-fire-monitor.git)
    cd forest-fire-monitor
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows [cite: 1131]
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add your NASA API key:
    ```text
    NASA_FIRMS_API_KEY=your_actual_api_key_here
    ```

## 🖥️ Running the System
[cite_start]The application requires running the backend and frontend concurrently in separate terminals[cite: 1139].

**Terminal 1: Backend API**
```bash
python backend/main.py
Runs at http://0.0.0.0:8000
```
**Terminal 2: Frontend Dashboard**
```bash
python streamlit run frontend/app.py
Runs at http://0.0.0.0:8501
```

## 📊 Model Performance
The XGBoost classification model achieved an overall accuracy of 99% on historical datasets. Performance was validated using precision, recall, and F1-score metrics.  

**Developer: Vaibhav Karbhantnal** Affiliation: Department of MCA, BMS Institute of Technology & Management Academic Year: 2024-25
