# This Streamlit application predicts heart disease risk using a trained ML model.
# It takes clinical parameters as input from the user.
# The prediction is based on the best selected Random Forest model.


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Config
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Heart Disease Prediction")
st.markdown("### AI-Based Heart Disease Risk Evaluation")
st.markdown("This system estimates the probability of heart disease based on clinical parameters.")

st.markdown("---")


# Load the trained machine learning model, scaler, and feature names.
model = joblib.load("best_heart_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")


# Input Section
st.header("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 100, 35,
                    help="Age in years. Risk increases with age.")

    sex = st.radio("Gender", ["Female", "Male"],
                   help="Males statistically have slightly higher risk.")

    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 110,
                         help="Normal: 90–120 mm Hg.")

    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 180,
                     help="Normal: Below 200 mg/dl.")

    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"],
                   help="High fasting sugar may indicate diabetes.")

with col2:
    st.header("🫀 Clinical Measurements")

    cp = st.selectbox("Chest Pain Type",
                      ["Typical Angina",
                       "Atypical Angina",
                       "Non-anginal Pain",
                       "Asymptomatic"])

    restecg = st.selectbox("Resting ECG",
                           ["Normal",
                            "ST-T Abnormality",
                            "Left Ventricular Hypertrophy"])

    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 170)

    exang = st.radio("Exercise Induced Angina", ["No", "Yes"])

    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 0.5)

    slope = st.selectbox("Slope of ST Segment",
                         ["Upsloping", "Flat", "Downsloping"])

    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])

    thal = st.selectbox("Thalassemia",
                        ["Normal", "Fixed Defect", "Reversible Defect"])

st.markdown("---")

# Encode Inputs
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

cp_dict = {"Typical Angina": 0, "Atypical Angina": 1,
           "Non-anginal Pain": 2, "Asymptomatic": 3}

restecg_dict = {"Normal": 0, "ST-T Abnormality": 1,
                "Left Ventricular Hypertrophy": 2}

slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

thal_dict = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

input_data = {
    "age": age,
    "sex": sex,
    "cp": cp_dict[cp],
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg_dict[restecg],
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope_dict[slope],
    "ca": ca,
    "thal": thal_dict[thal]
}

input_df = pd.DataFrame([input_data])
input_df = input_df[feature_names]
input_scaled = scaler.transform(input_df)

# Risk Category Function
def risk_category(prob):
    if prob <= 0.30:
        return "Low Risk", "🟢"
    elif prob <= 0.60:
        return "Moderate Risk", "🟡"
    elif prob <= 0.80:
        return "High Risk", "🟠"
    else:
        return "Critical Risk", "🔴"

# Prediction
if st.button("🔍 Assess Risk"):

    probability = model.predict_proba(input_scaled)[0]
    disease_prob = probability[0]
    category, icon = risk_category(disease_prob)

    st.header("🧾 Assessment Result")

    st.markdown(f"## {icon} {category}")
    st.write(f"### Heart Disease Probability: {disease_prob*100:.2f}%")

    if disease_prob > 0.80:
        st.error("Immediate cardiologist consultation recommended.")
    elif disease_prob > 0.60:
        st.warning("High risk detected. Medical advice recommended.")
    elif disease_prob > 0.30:
        st.info("Moderate risk. Lifestyle monitoring advised.")
    else:
        st.success("Low risk. Maintain healthy habits.")

    st.markdown("---")

    # Click-to-Open Medical Explanation

    with st.expander("🔎 View Medical Term Details"):
        st.markdown("""
        **Blood Pressure:**  
        Normal range: 90–120 mm Hg. High values strain the heart.

        **Cholesterol:**  
        Normal: Below 200 mg/dl. High cholesterol can block arteries.

        **Fasting Blood Sugar:**  
        Normal: Below 120 mg/dl. High values may indicate diabetes.

        **ECG:**  
        Measures heart's electrical activity.

        **Angina (Chest Pain):**  
        Pain caused by reduced blood flow to heart muscle.

        **Maximum Heart Rate:**  
        Higher response during exercise is generally healthy.

        **Oldpeak:**  
        Indicates heart stress during exercise.

        **Major Vessels (CA):**  
        Number of blocked heart arteries.

        **Thalassemia (Thal):**  
        Indicates blood flow defects in heart muscle.
        """)

    st.markdown("---")

    st.write("Always consult a qualified cardiologist for medical decisions.")

    st.markdown("---")

    # Patient Summary
    st.subheader("📋 Patient Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Age", "Gender", "Blood Pressure",
            "Cholesterol", "Max Heart Rate",
            "Chest Pain Type", "Major Vessels"
        ],
        "Value": [
            age,
            "Male" if sex == 1 else "Female",
            trestbps,
            chol,
            thalach,
            cp,
            ca
        ]
    })

    st.table(summary)