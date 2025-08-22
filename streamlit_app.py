import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Load Model + Scaler + Encoders
# -----------------------
model = joblib.load("svmmj_model.pkl")
scaler = joblib.load("scallerj.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Features used during training
selected_features = [
   'ID', 'State', 'LGA', 'Month', 'Year', 'Age', 'Gender', 'Fever',
   'Headache', 'Weakness', 'Malaise', 'Sore_Throat', 'Muscle_Pain',
   'Chest_Pain', 'Cough', 'Nausea', 'Vomiting', 'Diarrhea',
   'Abdominal_Pain', 'Facial_Swelling', 'Bleeding', 'Low_Blood_Pressure',
   'Hearing_Loss', 'Seizures', 'Tremors', 'Disorientation', 'Coma',
   'Shock', 'Pregnant', 'Hospitalized', 'Duration_of_Symptoms', 'Severity'
]

# -----------------------
# Streamlit App UI
# -----------------------
st.set_page_config(page_title="AI4Lassa Outbreak Predictor", layout="wide")

st.title("🦠 AI4Lassa: Outbreak Prediction App")
st.markdown("Upload patient health records and get **real-time outbreak predictions** using the trained SVM model.")

# File Upload
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head())

        # -----------------------
        # Align Columns with Training
        # -----------------------
        missing_cols = [col for col in selected_features if col not in df.columns]
        if missing_cols:
            st.error(f"❌ The uploaded file is missing required columns: {missing_cols}")
        else:
            features = df[selected_features].copy()

            # Encode categorical features
            for col, le in label_encoders.items():
                if col in features.columns:
                    try:
                        features[col] = le.transform(features[col].astype(str))
                    except ValueError as e:
                        st.error(f"Encoding error in column `{col}`: {e}")
                        st.stop()

            # Handle missing values
            features = features.fillna(0)

            # Scale
            features_scaled = scaler.transform(features)

            # -----------------------
            # Prediction
            # -----------------------
            predictions = model.predict(features_scaled)
            df["Predicted_Outcome"] = predictions

            # Show results
            st.subheader("📊 Prediction Results")
            st.dataframe(df[["ID", "State", "LGA", "Predicted_Outcome"]].head())

            # -----------------------
            # Visualization
            # -----------------------
            st.subheader("📈 Prediction Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x=predictions, palette="Set2", ax=ax)
            ax.set_title("Predicted Outbreak Cases")
            ax.set_xlabel("Outcome (0 = No Outbreak, 1 = Outbreak)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # Download
            st.subheader("💾 Download Results")
            st.download_button(
                label="Download Predictions as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="lassa_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
