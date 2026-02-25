import streamlit as st
import pandas as pd
import numpy as np
import joblib
import dask.dataframe as dd
import os

# Page Config
st.set_page_config(
    page_title="Big Data Healthcare Predictor",
    page_icon="🏥",
    layout="wide"
)

# Load Model & Scaler
@st.cache_resource
def load_ml_assets():
    model_path = 'outputs/big_data_model.pkl'
    scaler_path = 'outputs/big_data_scaler.pkl'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None

model, scaler = load_ml_assets()

# Header
st.title("🏥 Healthcare Big Data Analytics")
st.markdown("""
This dashboard demonstrates **Scalable Predictive Analytics**. 
The backend handles millions of patient records using **Dask** and **Parquet**, 
while the prediction service uses an incrementally trained **SGD Classifier**.

> [!WARNING]  
> **DISCLAIMER:** This application is based on **purely synthetic data** and is intended for **educational and demonstration purposes only**. It must not be used for clinical diagnosis or therapeutic decision-making.
""")

st.divider()

# Sidebar settings
st.sidebar.header("🔍 Patient Risk Prediction")
st.sidebar.info("Input patient data to estimate 30-day readmission risk.")

age = st.sidebar.slider("Age", 18, 100, 55)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 26.5)
glucose = st.sidebar.number_input("Glucose Level (mg/dL)", 50, 400, 110)
bp = st.sidebar.number_input("Blood Pressure (Systolic)", 80, 200, 125)

if st.sidebar.button("Predict Readmission"):
    if model and scaler:
        # Prepare input (must match 03_analysis.py features)
        # Features: ['Age', 'Gender_Num', 'BMI', 'Glucose_Level', 'Blood_Pressure']
        gender_num = 0 if gender == "Male" else 1
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender_Num': gender_num,
            'BMI': bmi,
            'Glucose_Level': glucose,
            'Blood_Pressure': bp
        }])
        
        # Scaling correctly
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        st.sidebar.divider()
        if prediction == 1:
            st.sidebar.error("⚠️ High Risk of Readmission")
        else:
            st.sidebar.success("✅ Low Risk of Readmission")
    else:
        st.sidebar.warning("Model/Scaler assets not found. Run analysis.py first.")

# --- MAIN CONTENT: DATA INSIGHTS ---
col1, col2 = st.columns(2)

FILENAME_PQ = "healthcare_big_data.parquet"

with col1:
    st.subheader("📊 Scalable Data Insights")
    if os.path.exists(FILENAME_PQ):
        # Use Dask for big data stats
        ddf = dd.read_parquet(FILENAME_PQ)
        total_rows = len(ddf)
        avg_bmi = ddf['BMI'].mean().compute()
        readmission_rate = ddf['Readmitted_Within_30_Days'].mean().compute() * 100
        
        st.metric("Total Records Analyzed", f"{total_rows:,}")
        m1, m2 = st.columns(2)
        m1.metric("Average BMI", f"{avg_bmi:.1f}")
        m2.metric("Readmission Rate", f"{readmission_rate:.1f}%")
    else:
        st.warning("Parquet data not found. Run generation script.")

with col2:
    st.subheader("🏗️ Architecture")
    st.info("""
    - **Format:** Partitioned Parquet (Columnar)
    - **Engine:** Dask (Distributed Out-of-Core Processing)
    - **ML Model:** Incremental SGD (Stochastic Gradient Descent)
    - **Optimization:** Column Pruning & Lazy Evaluation
    """)

# Data Sample
if os.path.exists(FILENAME_PQ):
    st.subheader("📄 Data Preview (Calculated via Dask)")
    sample_df = ddf.head()
    st.dataframe(sample_df)

st.divider()
st.caption("Developed for Big Data Healthcare Analytics Demo | Sebastian Lijewski, PhD")
