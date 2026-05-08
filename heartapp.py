
import streamlit as st
import pickle
import numpy as np

# Load model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("heart_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title(" Heart Disease Prediction App")

st.write("Enter patient details")

# Inputs
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 50, 250)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar >120", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 50, 250)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal", [0, 1, 2, 3])

# Prepare input
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

# Scale input
features = scaler.transform(features)

# Predict
if st.button("Predict"):

    result = model.predict(features)

    if result[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
