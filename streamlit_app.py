import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Load models and scaler (train and save them first)
kmeans = joblib.load('kmeans_model.pkl')
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Selected features (match your list)
selected_features = [
    'mean_spectral_centroid', 'std_spectral_centroid', 'mean_spectral_bandwidth', 'std_spectral_bandwidth',
    'mean_spectral_contrast', 'mean_spectral_flatness', 'mean_spectral_rolloff', 'zero_crossing_rate',
    'rms_energy', 'mean_pitch', 'min_pitch', 'max_pitch', 'std_pitch', 'spectral_skew',
    'spectral_kurtosis', 'energy_entropy', 'log_energy', 'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean'
]

# App structure
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "EDA & Visualization", "Classification and Clustering"])

if page == "Introduction":
    st.title("Human Voice Classification and Clustering")
    st.write("""
    **Overview**: This project classifies and clusters human voices using audio features. It preprocesses data, applies ML models, and provides real-time predictions via this interface.
    
    **Dataset**: Extracted features from voice recordings (e.g., spectral, pitch, MFCCs) with gender labels.
    
    **Objectives**: Preprocess data, cluster voices, classify gender, and deploy an interactive app.
    
    **Pipeline**: Data Prep → EDA → Model Training → Evaluation → Deployment.
    """)

elif page == "EDA & Visualization":
    st.title("EDA & Visualization")
    # Load dataset for viz (adjust path)
    df = pd.read_csv('voice_data.csv')
    
    st.subheader("Feature Distributions")
    fig, ax = plt.subplots(figsize=(10, 6))
    df[selected_features[:10]].hist(ax=ax)  # Sample 10 features
    st.pyplot(fig)
    
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[selected_features[:10] + ['label']].corr(), annot=True, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Pitch by Gender")
    fig, ax = plt.subplots()
    sns.boxplot(x='label', y='mean_pitch', data=df, ax=ax)
    st.pyplot(fig)

elif page == "Classification and Clustering":
    st.title("Classification and Clustering Predictions")
    st.write("Enter values for the top 20 features to get cluster assignment and gender classification.")
    
    # Input fields for features
    inputs = {}
    for feature in selected_features:
        inputs[feature] = st.number_input(f"{feature}", value=0.0, step=0.01)
    
    if st.button("Predict"):
        # Prepare input
        input_data = np.array([inputs[feat] for feat in selected_features]).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        
        # Clustering prediction
        cluster = kmeans.predict(input_scaled)[0]
        st.write(f"**Cluster Assignment**: {cluster} (e.g., 0 for one group, 1 for another)")
        
        # Classification prediction
        gender_pred = rf_model.predict(input_scaled)[0]
        gender_prob = rf_model.predict_proba(input_scaled)[0]
        gender_label = "Male" if gender_pred == 1 else "Female"
        st.write(f"**Gender Classification**: {gender_label} (Confidence: {max(gender_prob)*100:.2f}%)")
        
        # Interpretation
        st.subheader("Interpretation")
        st.write("- Clustering groups similar voices based on features.")
        st.write("- Classification predicts gender (Male=1, Female=0).")
        st.write("Note: Results are based on trained models; accuracy depends on data quality.")