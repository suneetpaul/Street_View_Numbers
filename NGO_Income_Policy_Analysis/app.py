import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Income Predictor · DeltaSquare NGO",
    page_icon="💼",
    layout="centered",
)

# ── White theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global white background */
    .stApp { background-color: #ffffff; color: #000000; }
    [data-testid="stSidebar"] { background-color: #f8f9fa; }

    /* All general text black */
    h1, h2, h3, h4, p, label, span, div { color: #000000; }

    /* Widget labels black */
    .stSlider label, .stSelectbox label,
    .stNumberInput label, .stCheckbox label,
    .stMarkdown p { color: #f8f9fa !important; }

    /* Header band */
    .header-band {
        background: linear-gradient(135deg, #2E6F9E 0%, #1a4a72 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(46,111,158,0.18);
    }
    .header-band h1 { margin: 0; font-size: 2rem; font-weight: 700; color: white !important; }
    .header-band p  { margin: 0.4rem 0 0; opacity: 0.88; font-size: 1rem; color: white !important; }

    /* Section cards */
    .card {
        background: #ffffff;
        border: 1.5px solid #e8edf2;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .card h3 { margin-top: 0; color: #2E6F9E !important; font-size: 1.05rem; }

    /* Section heading */
    .stMarkdown h3 { color: #ffffff !important; }

    /* Result boxes */
    .result-above {
        background: #eaf6ea;
        border-left: 5px solid #28a745;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        font-size: 1.25rem;
        font-weight: 600;
        color: #000000;
    }
    .result-below {
        background: #fff3e0;
        border-left: 5px solid #fd7e14;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        font-size: 1.25rem;
        font-weight: 600;
        color: #000000;
    }
    .prob-bar-label { font-size: 0.88rem; color: #000000; margin-bottom: 0.3rem; }

    /* Streamlit widget overrides for white look */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider { background-color: #ffffff !important; }

    /* Button */
    .stButton > button {
        background: #2E6F9E;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2.2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: background 0.2s;
        width: 100%;
    }
    .stButton > button:hover { background: #1a4a72; }

    /* Divider */
    hr { border-color: #e8edf2; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")
    #return joblib.load(r"D:\dripi\web_color\income-bracket-estimator\model.pkl")

try:
    model = load_model()
except Exception:
    st.error("⚠️  Could not load `random_forest_model.pkl`. Make sure it is in the same folder as `app.py`.")
    st.stop()

FEATURE_COLS = list(model.feature_names_in_)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="">
  <h1>💼 Income Prediction Tool</h1>
</div>
""", unsafe_allow_html=True)

# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown("### Enter Individual Details")

col1, col2 = st.columns(2)

with col1:
    
    age = st.slider("Age", min_value=17, max_value=90, value=35, step=1)
    sex = st.selectbox("Sex", ["male", "female"])
    race = st.selectbox("Race", [
        "White", "Black", "Asian-Pac-Islander", "Other"
    ])
    native_continent = st.selectbox("Native Continent", [
        "north_america", "europe", "south_america", "other"
    ], format_func=lambda x: x.replace("_", " ").title())

    has_capital_gain = st.checkbox("Has Capital Gain?")
    capital_gain = st.number_input("Capital Gain ($)", min_value=0, max_value=99999, value=0,
                                    disabled=not has_capital_gain)



    st.markdown('</div>', unsafe_allow_html=True)



with col2:
    
    education_years = st.slider("Years of Education", min_value=1, max_value=16, value=10)
    working_hours   = st.slider("Working Hours / Week", min_value=1, max_value=99, value=40)
    workclass = st.selectbox("Work Class", [
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Local-gov", "State-gov", "Without-pay"
    ])
    marital_status = st.selectbox("Marital Status", [
        "Married", "Never-married", "Not_married"
    ])
    occupation = st.selectbox("Occupation", [
        "Prof-specialty", "Exec-managerial", "Tech-support", "Sales",
        "Craft-repair", "Transport-moving", "Machine-op-inspct",
        "Handlers-cleaners", "Farming-fishing", "Protective-serv",
        "Other-service", "Priv-house-serv", "Armed-Forces"
    ])
    st.markdown('</div>', unsafe_allow_html=True)



# ── Build feature vector ───────────────────────────────────────────────────────
def build_features():
    row = {f: 0 for f in FEATURE_COLS}

    row["age"]                       = age
    row["education_no_of_years"]     = education_years
    row["working_hours_per_week"]    = working_hours
    row["has_capital_gain"]          = int(has_capital_gain)
    row["capital_gain"]              = capital_gain if has_capital_gain else 0
    

    # One-hot: sex (drop_first = Female baseline)
    if sex == "Male" and "sex_Male" in row:
        row["sex_Male"] = 1

    # One-hot: race (baseline = Amer-Indian-Eskimo)
    race_map = {
        "White":              "race_White",
        "Black":              "race_Black",
        "Asian-Pac-Islander": "race_Asian-Pac-Islander",
        "Other":              "race_Other",
    }
    key = race_map.get(race)
    if key and key in row:
        row[key] = 1

    # One-hot: workclass (baseline = Federal-gov)
    wc_key = f"workclass_{workclass}"
    if wc_key in row:
        row[wc_key] = 1

    # One-hot: marital_status (baseline = Married)
    if marital_status != "Married":
        ms_key = f"marital_status_{marital_status}"
        if ms_key in row:
            row[ms_key] = 1

    # One-hot: occupation (baseline = Adm-clerical)
    occ_key = f"occupation_{occupation}"
    if occ_key in row:
        row[occ_key] = 1

    # One-hot: native_contienent (baseline = africa/asia)
    nc_key = f"native_contienent_{native_continent}"
    if nc_key in row:
        row[nc_key] = 1

    return pd.DataFrame([row])[FEATURE_COLS]

# ── Predict button ─────────────────────────────────────────────────────────────
st.markdown("")
if st.button("🔍 Predict Income Class"):
    X_input = build_features()
    pred      = model.predict(X_input)[0]
    prob      = model.predict_proba(X_input)[0]  # [P(<=50K), P(>50K)]

    st.markdown("---")
    st.markdown("### 📊 Prediction Result")

    if pred == 1:
        st.markdown(
            '<div class="result-above">✅ Likely earns <b>more than $50K/year</b></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-below">⚠️ Likely earns <b>$50K/year or less</b></div>',
            unsafe_allow_html=True,
        )

    

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#000000; font-size:0.82rem;'>"
    "DeltaSquare NGO · Income Policy Analysis · Random Forest Classifier"
    "</p>",
    unsafe_allow_html=True,
)