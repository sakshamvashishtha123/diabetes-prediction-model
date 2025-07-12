import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit page config
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict the risk of diabetes:")

# Collect user input
preg = st.number_input("Number of Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
bp = st.number_input("Blood Pressure (mm Hg)", min_value=0)
skin = st.number_input("Skin Thickness (mm)", min_value=0)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0)
bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=1, step=1)

# Predict button
if st.button("Predict Diabetes Risk"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    # Scale the input using the same scaler used during training
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_input)

    # Output result
    if prediction[0] == 1:
        st.error("⚠️ You may have diabetes. Please consult a doctor.")
    else:
        st.success("✅ You are unlikely to have diabetes.")

