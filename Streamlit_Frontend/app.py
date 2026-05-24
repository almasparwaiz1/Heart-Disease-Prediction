import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ========================================== #
# 🛠️ MONKEY PATCH FOR SCIKIT-LEARN ERROR     #
# ========================================== #
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# Import BaseEstimator, TransformerMixin for CustomFeatureEngineer
from sklearn.base import BaseEstimator, TransformerMixin

# ========================================== #
# Define CustomFeatureEngineer Class         #
# (MUST be present when loading the pipeline)#
# ========================================== #
class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.risk_factor_cols = ['High_BP', 'High_Cholesterol', 'Diabetes', 'Smoking', 'Obesity', 'Family_History', 'Chronic_Stress']
        self.heart_symptoms = [
            'Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations', 'Dizziness',
            'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea'
        ]
        self.age_bins = [0, 40, 60, 84] # Using 84 as max age from EDA
        self.age_labels = ['Young', 'Middle_Aged', 'Elderly']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # 1. Age Groups
        X_copy['Age_Group'] = pd.cut(X_copy['Age'], bins=self.age_bins, labels=self.age_labels, right=False, include_lowest=True).astype(object)

        # 2. Risk Score Index
        existing_risk_factor_cols = [col for col in self.risk_factor_cols if col in X_copy.columns]
        X_copy['Risk_Score_Index'] = X_copy[existing_risk_factor_cols].sum(axis=1)

        # 3. High_BP_x_High_Cholesterol interaction
        X_copy['High_BP_x_High_Cholesterol'] = X_copy['High_BP'] * X_copy['High_Cholesterol']

        # 4. Heart Workload Index
        existing_heart_symptoms = [col for col in self.heart_symptoms if col in X_copy.columns]
        X_copy['Heart_Workload_Index'] = X_copy[existing_heart_symptoms].sum(axis=1)

        # 5. Age_x_High_Cholesterol interaction
        X_copy['Age_x_High_Cholesterol'] = X_copy['Age'] * X_copy['High_Cholesterol']

        return X_copy

# ========================================== #
# BASE DIRECTORY CONFIG                      #
# ========================================== #
BASE_DIR = "Streamlit_Frontend"
MODEL_PATH = os.path.join(BASE_DIR, "heart_risk_prediction_full_pipeline.joblib")

# ========================================== #
# PAGE CONFIG                                #
# ========================================== #
st.set_page_config(
    page_title="Heart Disease Risk AI",
    layout="wide",
    page_icon="❤️"
)

# ========================================== #
# CUSTOM CSS (CLINICAL 60-30-10 UI THEME)    #
# ========================================== #
st.markdown("""
<style>
    /* Main Background & Text Accessibility */
    .stApp {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }
    
    /* Primary Branding Headers (60%) */
    h1 {
        color: #0F2C59 !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #0F2C59;
        padding-bottom: 10px;
    }
    h2, h3 {
        color: #1E3A8A !important;
        font-weight: 600 !important;
    }
    
    /* Interactive Secondary Element Buttons (30%) */
    .stButton>button {
        background-color: #008080 !important;
        color: #FFFFFF !important;
        font-size: 16px !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 12px 28px !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(0, 128, 128, 0.2);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #06B6D4 !important;
        box-shadow: 0 6px 12px rgba(6, 182, 212, 0.4);
        transform: translateY(-1px);
    }
    
    /* Accent Elements & High Risk Indicators (10%) */
    .high-risk-alert {
        background-color: #FF6B6B !important;
        color: #FFFFFF !important;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
    }
    
    .warning-text {
        color: #F59E0B !important;
        font-weight: bold;
    }
    
    /* Clean Cards for Inputs */
    div[data-testid="stVerticalBlock"] > div {
        background-color: #FFFFFF;
    }
    
    /* Sidebar Styling */
    .stSidebar {
        background-color: #F3F4F6 !important;
        border-right: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# ========================================== #
# LOAD MODEL                                 #
# ========================================== #
@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

pipeline = load_pipeline()

if pipeline is None:
    st.stop()

# ========================================== #
# APP TITLE                                  #
# ========================================== #
st.title("❤️ Heart Disease Risk Prediction AI")
st.markdown("<p style='color: #4B5563; font-size: 1.1rem;'>AI-powered medical risk analysis built for clinical decision support and early detection.</p>", unsafe_allow_html=True)
st.markdown("---")


symptom_features = [
    'Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations',
    'Dizziness', 'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea'
]

risk_factor_features = [
    'High_BP', 'High_Cholesterol', 'Diabetes', 'Smoking',
    'Obesity', 'Sedentary_Lifestyle', 'Family_History', 'Chronic_Stress'
]

user_input = {}

# ========================================== #
# LAYOUT                                     #
# ========================================== #
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("👤 Patient Demographics")
    user_input["Age"] = st.slider("Age", 20, 90, 45)
    user_input["Gender"] = st.selectbox(
        "Gender",
        options=[("Female", 0), ("Male", 1)],
        format_func=lambda x: x[0]
    )[1]

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("⚕️ Clinical Symptoms")
    for f in symptom_features:
        user_input[f] = st.selectbox(
            f.replace("_", " "),
            options=[("No", 0.0), ("Yes", 1.0)],
            format_func=lambda x: x[0]
        )[1]

with col2:
    st.subheader("🏥 Medical & Lifestyle History")
    for f in risk_factor_features:
        user_input[f] = st.selectbox(
            f.replace("_", " "),
            options=[("No", 0.0), ("Yes", 1.0)],
            format_func=lambda x: x[0]
        )[1]

# ========================================== #
# PREDICTION                                 #
# ========================================== #
st.markdown("---")

st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Run Diagnostic Analysis")
st.markdown("</div>", unsafe_allow_html=True)

if predict_btn:
    input_df = pd.DataFrame([user_input])

    try:
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Diagnostic Assessment Result")

        if prediction == 1:
            st.markdown(
                f"<div class='high-risk-alert'>🚨 CRITICAL ALERT: HIGH RISK OF HEART DISEASE DETECTED</div>", 
                unsafe_allow_html=True
            )
            st.metric("Calculated Risk Probability", f"{probability[1] * 100:.1f}%")
            st.markdown("<p class='warning-text'>⚠️ Warning: Immediate clinical evaluation and diagnostic testing are strongly recommended.</p>", unsafe_allow_html=True)
        else:
            st.success("✅ PATIENT STATUS: LOW RISK DETECTED")
            st.metric("Confidence Probability", f"{probability[0] * 100:.1f}%")
            st.info("Recommendation: Maintain preventative lifestyle measures and follow standard periodic testing guidelines.")
            
    except Exception as e:
        st.error(f"An error occurred during pipeline prediction: {e}")

    st.markdown("---")
    st.caption("Disclaimer: This tool is an AI decision aid and should not replace professional medical diagnosis.")
