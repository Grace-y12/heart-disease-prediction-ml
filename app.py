import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and column list
model = joblib.load("src/heart_disease_model.pkl")
scaler = joblib.load("src/scaler.pkl")
expected_columns = joblib.load("src/columns.pkl")

# Streamlit UI
st.title("💓 Heart Disease Prediction App")
st.markdown("Fill in the patient information below and click **Predict** to assess heart disease risk.")

# Input form
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=130)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=250)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise-Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression", step=0.1, value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])

# Create input DataFrame
input_dict = {
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal],
}
input_df = pd.DataFrame(input_dict)

# One-hot encode and align columns
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

# Prediction trigger
if st.button("🔍 Predict"):
    # Scale and predict
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)

    # Display result
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ High chance of heart disease")
    else:
        st.success("✅ Low chance of heart disease")

