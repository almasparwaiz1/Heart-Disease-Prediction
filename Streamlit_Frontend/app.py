import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Page Configuration --- #
st.set_page_config(
    page_title="Clinical Heart Disease Risk Assessment",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed", # No sidebar by default
)

# --- Base Directory & File Paths --- #
BASE_DIR = r"F:\AI and Data Science Projects\Heart Disease Prediction app\Streamlit_Frontend"
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'stacking_classifier_model.pkl')
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, 'feature_columns.pkl')

# --- Inject Clinical Theme Custom CSS --- #
st.markdown(
    """
    <style>
        /* Base Application Layout Colors */
        :root {
            --primary: #0A2540;
            --secondary: #00A699;
            --accent: #FF5A5F;
        }
        
        /* Main Page Background & Headers */
        .main .block-container {
            padding-top: 2rem;
        }
        h1 {
            color: #0A2540 !important;
            font-weight: 700 !important;
        }
        h2, h3 {
            color: #0A2540 !important;
            font-weight: 600 !important;
        }
        
        /* Hide Sidebar Elements completely */
        [data-testid="stSidebar"] {
            display: none !important;
        }
        
        /* Button Styling (Teal / Aqua Primary Action) */
        div.stButton > button:first-child {
            background-color: #00A699 !important;
            color: white !important;
            border: none !important;
            padding: 0.6rem 2rem !important;
            font-weight: bold !important;
            border-radius: 6px !important;
            transition: all 0.3s ease;
            width: 100%;
        }
        div.stButton > button:first-child:hover {
            background-color: #00877d !important;
            box-shadow: 0px 4px 10px rgba(0, 166, 153, 0.3);
        }
        
        /* Custom Alerts / Card Indicators */
        .high-risk-alert {
            background-color: rgba(255, 90, 95, 0.1);
            border-left: 5px solid #FF5A5F;
            padding: 1.5rem;
            border-radius: 4px;
            color: #333;
            margin-bottom: 1rem;
        }
        .low-risk-alert {
            background-color: rgba(0, 166, 153, 0.1);
            border-left: 5px solid #00A699;
            padding: 1.5rem;
            border-radius: 4px;
            color: #333;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Model and Preprocessor --- #
@st.cache_resource(show_spinner="Loading predictive clinical models...")
def load_resources():
    try:
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        return scaler, model, feature_columns
    except FileNotFoundError as e:
        st.error(f"Critical System Error: Resource asset not found. Verification failed for: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected System Interruption during load initialization: {e}")
        st.stop()

scaler, model, feature_columns = load_resources()

# --- Feature Engineering and Preprocessing Function ---
# REMOVED @st.cache_data decorator here to allow dynamic feature engineering updates
def preprocess_input(input_df, feature_columns_list):
    processed_df = input_df.copy()

    # 1. Age Groups 
    bins = [0, 40, 60, 84] 
    labels = ['Young', 'Middle_Aged', 'Elderly']
    processed_df['Age_Group'] = pd.cut(processed_df['Age'], bins=bins, labels=labels, right=False, include_lowest=True)

    # 2. Risk Score Index
    risk_factor_cols = ['High_BP', 'High_Cholesterol', 'Diabetes', 'Smoking', 'Obesity', 'Family_History', 'Chronic_Stress']
    processed_df['Risk_Score_Index'] = processed_df[risk_factor_cols].sum(axis=1)

    # 3. High_BP_x_High_Cholesterol Interaction
    processed_df['High_BP_x_High_Cholesterol'] = processed_df['High_BP'] * processed_df['High_Cholesterol']

    # 4. Heart Workload Index
    heart_symptoms = [
        'Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations', 'Dizziness',
        'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea'
    ]
    processed_df['Heart_Workload_Index'] = processed_df[heart_symptoms].sum(axis=1)

    # 5. Age_x_High_Cholesterol Interaction
    processed_df['Age_x_High_Cholesterol'] = processed_df['Age'] * processed_df['High_Cholesterol']

    # One-hot encode 'Age_Group'
    processed_df = pd.get_dummies(processed_df, columns=['Age_Group'], drop_first=True)

    # Align columns with training structure
    final_df = pd.DataFrame(0, index=processed_df.index, columns=feature_columns_list)

    for col in final_df.columns:
        if col in processed_df.columns:
            final_df[col] = processed_df[col].astype(int) if processed_df[col].dtype == bool else processed_df[col]

    return final_df

# --- Prediction Function ---
def predict_risk(data_scaled):
    prediction = model.predict(data_scaled)
    prediction_proba = model.predict_proba(data_scaled)[:, 1]
    return prediction[0], prediction_proba[0]

# --- Main Dashboard Frame --- #
st.title("🩺 Clinical Heart Disease Risk Analytics")
st.markdown(
    "Welcome to the Diagnostic Support Dashboard. This system utilizes a trained "
    "Stacking Classifier ensemble architecture to evaluate risk metrics based on structured "
    "patient physiological indicators. Please input required metrics accurately below."
)
st.markdown("---")

# --- Clinical Forms Input Section --- #
st.subheader("Patient Clinical Profile Setup")

col1, col2 = st.columns(2)
user_input = {}

with col1:
    st.markdown("### Demographic & Acute Symptoms")
    user_input['Age'] = st.slider("Patient Age (Years)", min_value=18, max_value=100, value=50, step=1,
                                  help="Current verified physiological age.")
    user_input['Chest_Pain'] = st.slider("Chest Pain Severity / Presence", 0, 1, 0, format="%d",
                                         help="Is patient exhibiting signs of active chest pain? (0 = No, 1 = Yes)")
    user_input['Shortness_of_Breath'] = st.slider("Shortness of Breath (Dyspnea)", 0, 1, 0,
                                                help="Observable or reported breathing complications.")
    user_input['Fatigue'] = st.slider("Chronic Unexplained Fatigue", 0, 1, 0,
                                      help="Prolonged systemic exhaustion under standard workloads.")
    user_input['Palpitations'] = st.slider("Palpitations / Arrhythmia Symptoms", 0, 1, 0,
                                            help="Irregular or heavy pounding pulse sensations.")
    user_input['Dizziness'] = st.slider("Dizziness / Vertigo Instances", 0, 1, 0,
                                         help="Spells of loss of balance or persistent lightheadedness.")
    user_input['Swelling'] = st.slider("Peripheral Swelling (Edema)", 0, 1, 0,
                                        help="Noticeable fluid accumulation in limbs or joint regions.")
    user_input['Pain_Arms_Jaw_Back'] = st.slider("Referred Pain (Arms, Jaw, Back)", 0, 1, 0,
                                                 help="Radiating pain common with cardiovascular strain.")

with col2:
    st.markdown("### Chronic Risks & Clinical Baselines")
    user_input['Gender'] = st.slider("Gender Identity Mapping", 0, 1, 1, 
                                     help="Physiological Sex Category (0: Female, 1: Male).")
    user_input['Cold_Sweats_Nausea'] = st.slider("Cold Sweats / Nausea Presence", 0, 1, 0,
                                                    help="Concurrent standard acute cardiac stress indicators.")
    user_input['High_BP'] = st.slider("Hypertension History (High BP)", 0, 1, 0,
                                      help="Patient has documented or treated elevated arterial blood pressure.")
    user_input['High_Cholesterol'] = st.slider("Hyperlipidemia (High Cholesterol)", 0, 1, 0,
                                                help="Patient has historical serum lipid abnormalities.")
    user_input['Diabetes'] = st.slider("Diabetes Mellitus Diagnosis", 0, 1, 0,
                                        help="Documented glycemic regulation disorder history.")
    user_input['Smoking'] = st.slider("Tobacco / Nicotine Consumption", 0, 1, 0,
                                      help="Active regular or highly recent historical tobacco use.")
    user_input['Obesity'] = st.slider("Clinical Obesity Profile", 0, 1, 0,
                                      help="BMI indices falling in the range categorized as clinically obese.")
    user_input['Sedentary_Lifestyle'] = st.slider("Sedentary Activity Baseline", 0, 1, 0,
                                                   help="Lack of regular required structured metabolic physical exercise.")
    user_input['Family_History'] = st.slider("Family Genetic Cardiovascular History", 0, 1, 0,
                                                help="Direct biological lineage historical cardiovascular diagnoses.")
    user_input['Chronic_Stress'] = st.slider("Persistent Psychosocial / Chronic Stress", 0, 1, 0,
                                                help="Elevated cortisol/stress profiles over standard long durations.")

# Construct Vector Array via DataFrame
input_df = pd.DataFrame([user_input])

# --- Diagnostic Execution Block --- #
st.markdown("---")
if st.button("Execute Diagnostic Risk Assessment", help="Compute diagnostic probability arrays across classification boundaries."):
    with st.spinner("Processing analytical models across parameters..."):
        try:
            # Scale & Standardize Input Matrix
            processed_input_df = preprocess_input(input_df, feature_columns)
            data_scaled = scaler.transform(processed_input_df)
            data_scaled_df = pd.DataFrame(data_scaled, columns=feature_columns)

            # Compute Class Probability Outputs
            risk_prediction, risk_proba = predict_risk(data_scaled_df)

            st.markdown("## Analytical Outcome Breakdown")
            
            # Displays probability in a prominent metric format
            st.metric(label="Calculated Pathological Probability", value=f"{risk_proba:.2%}")

            if risk_prediction == 1:
                st.markdown(
                    f"""
                    <div class="high-risk-alert">
                        <h3>⚠️ Elevate Clinical Vigilance: High Cardiovascular Risk Profile</h3>
                        <strong>Clinical Directive:</strong> Immediate comprehensive cardiovascular diagnostics, laboratory evaluation, and specialized clinical consultation are highly indicated.
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="low-risk-alert">
                        <h3>✅ Low Evaluated Cardiovascular Risk Baseline</h3>
                        <strong>Clinical Directive:</strong> Maintain routine standard preventative lifestyle habits and maintain established scheduled follow-up evaluations.
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"Critical error executed within mathematical framework evaluation: {e}")