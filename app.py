import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Setup
st.set_page_config(page_title="Industrial AI Digital Twin", layout="wide")

st.title("🏭 Predictive Maintenance Dashboard")
st.markdown("### AI-Powered Condition Monitoring System")

# 2. Load Models (Cached so they don't reload every time you click)
@st.cache_resource
def load_models():
    # Adjust paths if your models are in a different folder
    try:
        classifier = tf.keras.models.load_model('../models/bearing_cnn_model.keras')
        regressor = tf.keras.models.load_model('../models/bearing_rul_model.keras')
        return classifier, regressor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

classifier_model, rul_model = load_models()

# 3. File Uploader
st.sidebar.header("Input Data")
uploaded_file = st.sidebar.file_uploader("Upload Sensor Data")

if uploaded_file is not None:
    # --- A. PREPROCESSING ---
    st.sidebar.success("File Uploaded Successfully")
    
    # Read the raw file
    # (Assuming tab-separated values like NASA dataset)
    df = pd.read_csv(uploaded_file, sep='\t', header=None)
    df.columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']
    
    # Show Raw Data Preview
    with st.expander("View Raw Sensor Data"):
        st.dataframe(df.head())

    # PREPARE DATA FOR AI (CRITICAL STEP)
    # We must match the "Training Shape" exactly.
    # 1. Downsample (Take every 10th point) -> 20,480 becomes 2,048
    df_downsampled = df.iloc[::10]
    
    # 2. Format for Classifier (Needs all 4 sensors)
    # Shape: (1, 2048, 4) -> The '1' is the batch size
    X_classifier = df_downsampled.values.reshape(1, 2048, 4)
    
    # 3. Format for RUL Regressor (Needs only Bearing 1)
    # Shape: (1, 2048, 1)
    X_regressor = df_downsampled['Bearing 1'].values.reshape(1, 2048, 1)

    # --- B. AI ANALYSIS ---
    col1, col2 = st.columns(2)
    
    # Run Classifier
    prediction_prob = classifier_model.predict(X_classifier, verbose=0)[0][0]
    
    # Interpret Result (Threshold 0.5)
    is_faulty = prediction_prob > 0.5
    
    with col1:
        st.subheader("Machine Status")
        if is_faulty:
            st.error("🚨 CRITICAL FAULT DETECTED")
            st.metric(label="Confidence", value=f"{prediction_prob*100:.2f}%")
        else:
            st.success("✅ SYSTEM HEALTHY")
            st.metric(label="Confidence", value=f"{(1-prediction_prob)*100:.2f}%")

    with col2:
        st.subheader("Prognostics (RUL)")
        if is_faulty:
            # Only run the heavy regressor if there is a fault!
            rul_prediction = rul_model.predict(X_regressor, verbose=0)[0][0]
            
            # rul_prediction is between 0.0 (Dead) and 1.0 (New)
            life_percentage = rul_prediction * 100
            
            # Create a progress bar
            st.progress(int(max(0, min(100, life_percentage))))
            st.metric(label="Remaining Useful Life", value=f"{life_percentage:.1f}%")
            
            if life_percentage < 10:
                st.warning("⚠️ MAINTENANCE REQUIRED IMMEDIATELY")
        else:
            st.info("RUL Analysis not required for healthy systems.")

    # --- C. VISUALIZATION ---
    st.markdown("---")
    st.subheader("Real-Time Vibration Analysis")
    
    # Plot the 4 sensors
    fig, ax = plt.subplots(figsize=(10, 3))
    for col in df_downsampled.columns:
        ax.plot(df_downsampled[col], label=col, alpha=0.7)
    
    ax.set_title("Vibration Signature (Downsampled)")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Please upload a vibration file to begin analysis.")