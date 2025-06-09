
import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("heart_model.pkl")

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("🫀 Heart Disease Risk Predictor")
st.markdown("Enter your health parameters to check your risk of heart disease.")

with st.form("input_form"):
    age = st.slider("Age", 20, 80, 50, help="Age in years. Risk increases with age.")
    sex = st.radio("Sex", ["Female", "Male"], help="Biological sex. Males may have higher early-life risk.")
    sex = 0 if sex == "Female" else 1

    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic (No Pain)": 3
    }
    cp = st.selectbox("Chest Pain Type", list(cp_map.keys()), help="Chest pain type: Typical, Atypical, Non-anginal, or Asymptomatic.")
    cp = cp_map[cp]

    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120, help="Blood pressure while resting. High BP is a major risk factor.")
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200, help="Cholesterol level. High values can indicate heart risk.")

    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], help="Blood sugar level after fasting. Can indicate diabetes.")
    fbs = 1 if fbs == "Yes" else 0

    restecg_map = {
        "Normal": 0,
        "ST-T Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    restecg = st.selectbox(
        "Resting ECG Results (electrocardiogram)",
        list(restecg_map.keys()),
        help="Normal: no abnormalities, ST-T: possible ischemia, LVH: thickened heart muscle"
    )
    restecg = restecg_map[restecg]

    thalach = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, 150, help="Peak exercise heart rate. Lower values could suggest cardiovascular stress.")

    exang = st.radio("Exercise Induced Angina (chest pain)", ["No", "Yes"], help="Chest pain caused by exercise, often a symptom of blocked arteries.")
    exang = 1 if exang == "Yes" else 0

    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, help="ST segment depression during exercise. Higher values = higher risk.")

    slope_map = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    slope = st.selectbox("Slope of ST Segment", list(slope_map.keys()), help="Trend of ST segment on ECG. Flat/downsloping can indicate abnormality.")
    slope = slope_map[slope]

    ca_map = {
        "None": 0,
        "One": 1,
        "Two": 2,
        "Three": 3
    }
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", list(ca_map.keys()), help="Count of major vessels visible under imaging. Higher = worse.")
    ca = ca_map[ca]

    thal_map = {
        "Normal": 1,
        "Fixed Defect": 2,
        "Reversible Defect": 3
    }
    thal = st.selectbox("Thalassemia Type", list(thal_map.keys()), help="Blood disorder type affecting oxygen flow. Some types increase heart risk.")
    thal = thal_map[thal]

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)[0]

    # Fix for IndexError in predict_proba
    proba_array = model.predict_proba(input_data)[0]
    if len(proba_array) == 2:
        proba = proba_array[1]
    else:
        proba = proba_array[0]

    # Display readable input summary
    summary_df = pd.DataFrame([{
        "Age": age,
        "Sex": "Female" if sex == 0 else "Male",
        "Chest Pain": [k for k, v in cp_map.items() if v == cp][0],
        "Resting BP": trestbps,
        "Cholesterol": chol,
        "FBS": "Yes" if fbs == 1 else "No",
        "Rest ECG": [k for k, v in restecg_map.items() if v == restecg][0],
        "Max HR": thalach,
        "Exercise Angina": "Yes" if exang == 1 else "No",
        "Oldpeak": oldpeak,
        "Slope": [k for k, v in slope_map.items() if v == slope][0],
        "CA": [k for k, v in ca_map.items() if v == ca][0],
        "Thal": [k for k, v in thal_map.items() if v == thal][0]
    }])
    st.subheader("📝 Your Input Summary")
    st.dataframe(summary_df)

    # Display result
    if proba >= 0.7:
        st.error(f"🚨 High risk of heart disease! ({proba*100:.2f}% confidence)")
    elif proba >= 0.4:
        st.warning(f"⚠️ Moderate risk of heart disease ({proba*100:.2f}% confidence)")
    else:
        st.success(f"✅ Low risk of heart disease ({proba*100:.2f}% confidence)")
