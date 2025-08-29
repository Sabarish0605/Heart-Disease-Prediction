import streamlit as st
import pandas as pd
import pickle

# Load the trained ensemble model
with open(r'c:\Users\sabar\Desktop\Disease predicition\Heart-disease-prediction\ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Heart Disease Prediction")

# Input fields for features
age = st.number_input('Age', min_value=1, max_value=120, value=50)
cholesterol = st.number_input('Cholesterol', min_value=100, max_value=600, value=200)
bp = st.number_input('Blood Pressure', min_value=50, max_value=200, value=120)

sex = st.selectbox('Sex', options=['Male', 'Female'])
diabetes = st.selectbox('Diabetes', options=['No', 'Yes'])

# Encode categorical inputs to match training data encoding
sex_female = 1 if sex == 'Female' else 0
sex_male = 1 if sex == 'Male' else 0
diabetes_no = 1 if diabetes == 'No' else 0
diabetes_yes = 1 if diabetes == 'Yes' else 0

# Prepare input dataframe with exact feature names used during training
input_df = pd.DataFrame({
    'age': [age],
    'cholesterol': [cholesterol],
    'bp': [bp],
    'sex_Female': [sex_female],
    'sex_Male': [sex_male],
    'diabetes_No': [diabetes_no],
    'diabetes_Yes': [diabetes_yes],
    'age_cholesterol': [age * cholesterol],
    'bp_cholesterol': [bp * cholesterol]
})

threshold = st.slider('Prediction Threshold', 0.0, 1.0, 0.5, 0.01)

if st.button('Predict'):
    proba = model.predict_proba(input_df)[0][1]  # Probability of class 1
    prediction = 1 if proba >= threshold else 0
    st.write(f"Prediction probability for heart disease: {proba:.2f}")
    if prediction == 1:
        st.error('The model predicts presence of heart disease.')
    else:
        st.success('The model predicts no heart disease.')