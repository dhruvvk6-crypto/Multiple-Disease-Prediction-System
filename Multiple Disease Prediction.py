# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 01:35:45 2025

@author: Lenovo
"""

import numpy as np
import pickle
import streamlit as st
import os
from streamlit_option_menu import option_menu

# =======================
# Load the Saved Models
# =======================

import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'diabetes_model.sav'), 'rb') as f:
    diabetes_model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'heart_model.sav'), 'rb') as f:  # use the model you trained earlier
    heart_model = pickle.load(f)


with open(os.path.join(BASE_DIR, 'covid_prediction_model.sav'), 'rb') as f:
    covid_model = pickle.load(f)



# =======================
# Sidebar Navigation
# =======================
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Covid 19 Prediction'],
        icons=['activity', 'heart', 'mask'],
        default_index=0
    )


# =======================
# Diabetes Prediction Page
# =======================
if selected == 'Diabetes Prediction':

    st.title('🩸 Diabetes Prediction System')

    # Input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    # Prediction
    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            diab_prediction = diabetes_model.predict([[float(Pregnancies), float(Glucose), float(BloodPressure),
                                                       float(SkinThickness), float(Insulin), float(BMI),
                                                       float(DiabetesPedigreeFunction), float(Age)]])
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is Diabetic 😟'
            else:
                diab_diagnosis = 'The person is Not Diabetic 😊'
        except:
            diab_diagnosis = '⚠️ Please enter valid numeric values.'

    st.success(diab_diagnosis)


# =======================
# =======================
# Heart Disease Prediction Page
# =======================
if selected == 'Heart Disease Prediction':

    st.title('❤️ Heart Disease Prediction System')

    # --- User Input ---
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, step=1)
        resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=250, step=1)
        restecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
        st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

    with col2:
        sex = st.selectbox('Sex', ['M', 'F'])
        cholesterol = st.number_input('Cholesterol (mg/dl)', min_value=100, max_value=600, step=1)
        max_hr = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=250, step=1)
        oldpeak = st.number_input('ST Depression induced by exercise', min_value=0.0, max_value=10.0, step=0.1)

    with col3:
        chest_pain = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
        fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        exercise_angina = st.selectbox('Exercise Induced Angina', ['N', 'Y'])

    # --- Mappings (Convert to numeric values for the model) ---
    sex_map = {'M': 1, 'F': 0}
    cp_map = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
    restecg_map = {'Normal': 0, 'ST': 1, 'LVH': 2}
    exang_map = {'N': 0, 'Y': 1}
    fbs_map = {'No': 0, 'Yes': 1}
    slope_map = {'Up': 0, 'Flat': 1, 'Down': 2}

    # --- Prepare input for prediction ---
    heart_features = [
        age,
        sex_map[sex],
        cp_map[chest_pain],
        resting_bp,
        cholesterol,
        fbs_map[fasting_bs],
        restecg_map[restecg],
        max_hr,
        exang_map[exercise_angina],
        oldpeak,
        slope_map[st_slope]
    ]

    heart_diagnosis = ''

    # --- Predict ---
    if st.button('Heart Disease Test Result'):
        try:
            heart_prediction = heart_model.predict([heart_features])

            if heart_prediction[0] == 1:
                heart_diagnosis = '💔 The person is likely to have Heart Disease.'
            else:
                heart_diagnosis = '💚 The person is unlikely to have Heart Disease.'

        except Exception as e:
            heart_diagnosis = f'⚠️ Error: {e}'

    st.success(heart_diagnosis)


# =======================
# Covid-19 Prediction Page 
# =======================
if selected == 'Covid 19 Prediction':

    st.title('😷 COVID-19 Prediction System')

    st.markdown("Enter 1 for **Yes**, and 0 for **No** in all the fields below:")

    col1, col2, col3 = st.columns(3)

    with col1:
        Breathing_Problem = st.text_input('Breathing Problem (1/0)')
        Fever = st.text_input('Fever (1/0)')
        Dry_Cough = st.text_input('Dry Cough (1/0)')
        Sore_Throat = st.text_input('Sore Throat (1/0)')
        Running_Nose = st.text_input('Running Nose (1/0)')
        Asthma = st.text_input('Asthma (1/0)')
        Chronic_Lung_Disease = st.text_input('Chronic Lung Disease (1/0)')

    with col2:
        Headache = st.text_input('Headache (1/0)')
        Heart_Disease = st.text_input('Heart Disease (1/0)')
        Diabetes = st.text_input('Diabetes (1/0)')
        Hyper_Tension = st.text_input('Hyper Tension (1/0)')
        Fatigue = st.text_input('Fatigue (1/0)')
        Gastrointestinal = st.text_input('Gastrointestinal (1/0)')

    with col3:
        Abroad_travel = st.text_input('Abroad Travel (1/0)')
        Contact_with_COVID_Patient = st.text_input('Contact with COVID Patient (1/0)')
        Attended_Large_Gathering = st.text_input('Attended Large Gathering (1/0)')
        Visited_Public_Exposed_Places = st.text_input('Visited Public Exposed Places (1/0)')
        Family_working_in_Public_Exposed_Places = st.text_input('Family working in Public Exposed Places (1/0)')
        Wearing_Masks = st.text_input('Wearing Masks (1/0)')
        Sanitization_from_Market = st.text_input('Sanitization from Market (1/0)')

    covid_diagnosis = ''

    if st.button('Covid Test Result'):
        try:
            covid_prediction = covid_model.predict([[ 
                float(Breathing_Problem), float(Fever), float(Dry_Cough), float(Sore_Throat),
                float(Running_Nose), float(Asthma), float(Chronic_Lung_Disease), float(Headache),
                float(Heart_Disease), float(Diabetes), float(Hyper_Tension), float(Fatigue),
                float(Gastrointestinal), float(Abroad_travel), float(Contact_with_COVID_Patient),
                float(Attended_Large_Gathering), float(Visited_Public_Exposed_Places),
                float(Family_working_in_Public_Exposed_Places), float(Wearing_Masks),
                float(Sanitization_from_Market)
            ]])

            if covid_prediction[0] == 1:
                covid_diagnosis = '😷 The person is likely COVID-19 Positive'
            else:
                covid_diagnosis = '😊 The person is likely COVID-19 Negative'

        except Exception as e:
            covid_diagnosis = f'⚠️ Please enter valid numeric values.\n\nError: {e}'

    st.success(covid_diagnosis)
