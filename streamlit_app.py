import streamlit as st
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Load scaler and model
scaler = pickle.load(open("scallerj.pkl", "rb"))
svm_model = pickle.load(open("svmmj_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
# Streamlit UI
st.set_page_config(page_title="AI4Lassa Prediction App", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .title {text-align: center; font-size: 36px; color: #2E86C1; font-weight: bold;}
    .subtitle {text-align: center; font-size: 18px; color: #566573;}
    .result-card {padding:20px; border-radius:10px; background-color:#D5F5E3; text-align:center;}
    .error-card {padding:20px; border-radius:10px; background-color:#FADBD8; text-align:center;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🦠 AI4Lassa Outbreak Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload patient data to predict Lassa fever outcome</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info("This app uses a trained SVM model to predict **Lassa fever outcomes** "
                "based on clinical and demographic features.")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Show dataset summary
    with st.expander("🔎 Dataset Summary"):
        st.write(df.describe(include="all"))

        # Plot Age distribution
        fig, ax = plt.subplots()
        sns.histplot(df["Age"], bins=10, kde=True, ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)

    # Predict Button
    if st.button("🚀 Run Prediction"):
        with st.spinner("Processing data..."):
            time.sleep(2)  # Simulate processing

            # Drop non-numeric features before scaling (make sure you encoded them before training)
            features = df.drop(columns=["Outcome"], errors="ignore")

            # Scale numeric features
            features_scaled = scaler.transform(features)

            # Predict
            predictions = svm_model.predict(features_scaled)
            df["Predicted_Outcome"] = predictions

        # Show Results
        st.success("✅ Prediction Completed!")

        # Display result in a card
        if df["Predicted_Outcome"].sum() > 0:
            st.markdown('<div class="result-card"><h3>⚠️ High Risk Cases Detected</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-card"><h3>✅ No High Risk Cases</h3></div>', unsafe_allow_html=True)

        st.write("### 📝 Prediction Results")
        st.dataframe(df)

        # Download option
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="lassa_predictions.csv",
            mime="text/csv",
        )
