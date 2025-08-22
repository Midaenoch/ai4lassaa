import streamlit as st
import pandas as pd
import pickle

# Load pickled model and scaler
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Define input columns
columns = [
    "ID","State","LGA","Month","Year","Age","Gender","Fever","Headache",
    "Weakness","Malaise","Sore_Throat","Muscle_Pain","Chest_Pain","Cough",
    "Nausea","Vomiting","Diarrhea","Abdominal_Pain","Facial_Swelling",
    "Bleeding","Low_Blood_Pressure","Hearing_Loss","Seizures","Tremors",
    "Disorientation","Coma","Shock","Pregnant","Hospitalized",
    "Duration_of_Symptoms"
]

# Streamlit form
st.title("AI4Lassa - Outbreak Prediction")

# Collect user input
input_data = {}
for col in columns:
    if col in ["State", "LGA", "Gender", "Month"]:  # categorical inputs
        input_data[col] = st.text_input(f"Enter {col}")
    elif col in ["Fever","Headache","Weakness","Malaise","Sore_Throat","Muscle_Pain",
                 "Chest_Pain","Cough","Nausea","Vomiting","Diarrhea","Abdominal_Pain",
                 "Facial_Swelling","Bleeding","Low_Blood_Pressure","Hearing_Loss",
                 "Seizures","Tremors","Disorientation","Coma","Shock","Pregnant","Hospitalized"]:
        input_data[col] = st.selectbox(f"{col}", ["Yes", "No"])
    else:
        input_data[col] = st.number_input(f"Enter {col}", min_value=0)

# Convert to DataFrame
df = pd.DataFrame([input_data])

# Encode categorical variables
for col, encoder in encoders.items():
    if col in df.columns:
        df[col] = encoder.transform(df[col])

# Drop ID (not used in prediction)
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

# Scale numerical features
X_scaled = scaler.transform(df)

# Prediction
if st.button("Predict"):
    prediction = model.predict(X_scaled)[0]
    st.success(f"Predicted Outcome: {prediction}")
