import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoders
model = joblib.load("lassa_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🦠 AI4Lassa Outbreak Prediction System")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # =======================
    # FILE UPLOAD MODE
    # =======================
    data = pd.read_csv(uploaded_file)

    # Encode categorical features using fitted encoders
    for col, le in label_encoders.items():
        if col in data.columns:
            data[col] = le.transform(data[col])

    # Scale features
    X_scaled = scaler.transform(data)

    # Predict
    predictions = model.predict(X_scaled)

    # Add results to dataframe
    data["Prediction"] = predictions

    # Decode back to readable values
    for col, le in label_encoders.items():
        if col in data.columns:
            data[col] = le.inverse_transform(data[col])

    st.subheader("📊 Prediction Results (Uploaded CSV)")
    st.dataframe(data)

    # Download button
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="lassa_predictions.csv",
        mime="text/csv"
    )

else:
    # =======================
    # MANUAL ENTRY MODE
    # =======================
    st.subheader("📝 Enter Case Data Manually")

    # Dropdown options from encoders
    state_options = label_encoders["State"].classes_
    lga_options = label_encoders["LGA"].classes_
    gender_options = label_encoders["Gender"].classes_
    severity_options = label_encoders["Severity"].classes_
    outcome_options = label_encoders["Outcome"].classes_

    # Form inputs
    state = st.selectbox("Select State", state_options)
    lga = st.selectbox("Select LGA", lga_options)
    gender = st.selectbox("Select Gender", gender_options)
    severity = st.selectbox("Select Severity", severity_options)
    outcome = st.selectbox("Select Outcome", outcome_options)
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    cases = st.number_input("Number of Cases", min_value=0, step=1)

    # Submit button
    if st.button("Predict Outbreak Risk"):
        input_data = pd.DataFrame([{
            "State": state,
            "LGA": lga,
            "Gender": gender,
            "Severity": severity,
            "Outcome": outcome,
            "Age": age,
            "Cases": cases
        }])

        # Encode categorical features
        for col, le in label_encoders.items():
            if col in input_data.columns:
                input_data[col] = le.transform(input_data[col])

        # Scale features
        X_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(X_scaled)[0]

        st.success(f"Predicted Outbreak Risk: **{prediction}**")
