import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
with open("models/svmm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaller.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🧪 AI4Lassa Fever Outbreak Prediction App")
st.markdown("Upload your data file (CSV or Excel) or manually input data to predict potential Lassa Fever outbreaks.")

# Define expected features
selected_features = [
    'Cases', 'Any_Confirmed_Cases', 'Reports_All', 'LGA_Mean_Cases',
    'Cases_SuspectedUnconfirmed', 'Year', 'NumDiagCentres', 'LabDist',
    'Source', 'TotalPopulation_ByYear', 'LabTravelTime', 'AgriProp_ESA',
    'ForestProp_ESA', 'UrbanProp_ESA', 'TempMeanAnnual_201119_NOAA',
    'CHELSA_PrecipTotalAnnual', 'TotalRuralPop2015', 'CHELSA_TempAnnualMean',
    'CHELSA_PrecipWettestQ'
]

# 1. File upload
uploaded_file = st.file_uploader("📤 Upload CSV or Excel file", type=["csv", "xlsx"])

# 2. Manual fallback
manual_input = {}

if st.button("Predict"):
    try:
        if uploaded_file is not None:
            # Read file using pandas
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Validate required columns
            missing_cols = [col for col in selected_features if col not in df.columns]
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
                st.stop()

            input_data = df[selected_features].copy()

        else:
            input_data = pd.DataFrame([manual_input])

        # Make predictions
        scaled_input = scaler.transform(input_data)
        predictions = model.predict(scaled_input)

        # Add prediction info
        input_data["Prediction"] = predictions
        input_data["Status"] = input_data["Prediction"].apply(lambda x: "🦠 Outbreak" if x == 1 else "✅ No Outbreak")
        input_data["Recommendation"] = input_data["Prediction"].apply(
            lambda x: "Alert Health Authorities" if x == 1 else "Continue Monitoring"
        )

        # Optional state label
        input_data.insert(0, "State", [f"State {i+1}" for i in range(len(input_data))])

        # Display table
        st.markdown("### 📊 Prediction Results")
        st.dataframe(input_data[["State", "Status", "Recommendation"]])

        # Download button
        csv = input_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", data=csv, file_name="lassa_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
