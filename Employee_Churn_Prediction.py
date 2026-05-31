import streamlit as st
import pandas as pd
from joblib import load
import dill

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title='Employee Churn Prediction', layout='centered')

# ── Custom CSS styling ────────────────────────────────────────────────────────
st.markdown("""
    <style>
        /* Main title */
        h1 {
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 2.2rem;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 0rem;
        }
        /* Subheaders (Categorical / Numerical sections) */
        h2 {
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 1.2rem;
            font-weight: 600;
            color: #16213e;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 4px;
        }
        /* Input field labels */
        label {
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            color: #444444 !important;
        }
        /* Select box and number input height */
        .stSelectbox > div > div,
        .stNumberInput > div > div > input {
            min-height: 42px;
            font-size: 0.95rem;
        }
        /* Predict button */
        .stButton > button {
            background-color: #1a1a2e;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            margin-top: 1rem;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #16213e;
            border-color: #16213e;
        }
        /* Prediction result text */
        .prediction-box {
            background-color: #f0f4ff;
            border-left: 4px solid #1a1a2e;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 1rem;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ── Load the pretrained model ─────────────────────────────────────────────────
with open('pipeline.pkl', 'rb') as file:
    model = dill.load(file)

my_feature_dict = load('my_feature_dict.pkl')


# ── Function to predict churn ─────────────────────────────────────────────────
def predict_churn(data):
    prediction = model.predict(data)
    return prediction


# ── App Header ────────────────────────────────────────────────────────────────
st.title('Employee Churn Prediction App')
st.subheader('Based on Employee Dataset')

# ── Categorical Features ──────────────────────────────────────────────────────
st.subheader('Categorical Features')
categorical_input = my_feature_dict.get('CATEGORICAL')
categorical_input_vals = {}

for i, col in enumerate(categorical_input.get('Column Name').values()):
    categorical_input_vals[col] = st.selectbox(col, categorical_input.get('Members')[i])

# ── Numerical Features ────────────────────────────────────────────────────────
st.subheader('Numerical Features')
numerical_input = my_feature_dict.get('NUMERICAL')
numerical_input_vals = {}

for col in numerical_input.get('Column Name'):
    numerical_input_vals[col] = st.number_input(col, step=1, value=0)

# ── Combine inputs ────────────────────────────────────────────────────────────
input_data = dict(list(categorical_input_vals.items()) + list(numerical_input_vals.items()))
input_data = pd.DataFrame.from_dict(input_data, orient='index').T

# ── Churn Prediction ──────────────────────────────────────────────────────────
if st.button('Predict'):
    prediction = predict_churn(input_data)[0]
    translation_dict = {'Leave': 'Expected', 'Stay': 'Not Expected'}
    prediction_translate = translation_dict.get(prediction)
    st.markdown(
        f'<div class="prediction-box">The Prediction is <b>{prediction}</b>, '
        f'Hence Employee is <b>{prediction_translate}</b> to Churn.</div>',
        unsafe_allow_html=True
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('---')
st.subheader('Created by Safeer')
