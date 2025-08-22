import streamlit as st
import pandas as pd
import pickle
import time
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open("Svmmj_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scallerj.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="AI4Lassa Outbreak Prediction", layout="wide")

st.title("🦠 AI4Lassa Fever Outbreak Prediction System")
st.markdown("Upload your dataset to get predictions on possible **Lassa Fever outbreaks**.")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(data.head())

        # Features (remove target if present)
        if "Outcome" in data.columns:
            features = data.drop(columns=["Outcome"])
        else:
            features = data

        # Scale the features
        features_scaled = scaler.transform(features)

        # Prediction button with timer
        if st.button("🔮 Predict Outbreaks"):
            start_time = time.time()
            predictions = model.predict(features_scaled)
            end_time = time.time()

            # Convert predictions to readable labels
            results = ["Outbreak" if p == 1 else "No Outbreak" for p in predictions]

            # Add results to dataframe
            data["Prediction"] = results

            st.subheader("✅ Prediction Results")
            st.dataframe(data)

            st.success(f"Predictions completed in {end_time - start_time:.2f} seconds ⏱")

            # Show summary
            outbreak_count = results.count("Outbreak")
            no_outbreak_count = results.count("No Outbreak")

            st.info(f"**Summary:** Outbreaks = {outbreak_count}, No Outbreaks = {no_outbreak_count}")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
