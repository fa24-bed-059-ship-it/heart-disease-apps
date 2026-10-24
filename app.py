"""CardioScope - a small Streamlit app that estimates heart disease risk
from clinical measurements, using a Random Forest trained on the UCI
Cleveland dataset. Educational use only."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "cardio_model.joblib"

st.set_page_config(page_title="CardioScope", page_icon="🫀", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


bundle = load_bundle()
pipeline = bundle["pipeline"]
FEATURES = bundle["features"]
metrics = bundle["metrics"]

# ---------- label maps ----------
CHEST_PAIN = {
    "Typical angina": 1,
    "Atypical angina": 2,
    "Non-anginal pain": 3,
    "No symptoms": 4,
}
ECG = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2,
}
SLOPE = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
THAL = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}
NICE_NAMES = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar > 120",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression",
    "slope": "ST slope",
    "ca": "Major vessels coloured",
    "thal": "Thalassemia result",
}

# ---------- sidebar: inputs ----------
st.sidebar.title("🫀 CardioScope")
st.sidebar.caption("Fill in the details, then read the result on the right.")

with st.sidebar.expander("Basics", expanded=True):
    age = st.slider("Age", 20, 90, 54)
    sex_label = st.radio("Sex", ["Female", "Male"], horizontal=True, index=1)
    chest_label = st.selectbox("Chest pain type", list(CHEST_PAIN))

with st.sidebar.expander("Vitals and blood work", expanded=True):
    trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 200, 130)
    chol = st.slider("Cholesterol (mg/dl)", 100, 570, 245)
    fbs_high = st.checkbox("Fasting blood sugar above 120 mg/dl")

with st.sidebar.expander("Stress test and imaging", expanded=False):
    ecg_label = st.selectbox("Resting ECG", list(ECG))
    thalach = st.slider("Maximum heart rate reached", 70, 210, 150)
    exang = st.checkbox("Chest pain during exercise")
    oldpeak = st.slider("ST depression after exercise", 0.0, 6.5, 1.0, step=0.1)
    slope_label = st.selectbox("Slope of peak-exercise ST segment", list(SLOPE), index=1)
    ca = st.select_slider("Major vessels coloured by fluoroscopy", options=[0, 1, 2, 3])
    thal_label = st.selectbox("Thalassemia test result", list(THAL))

patient = {
    "age": age,
    "sex": 1 if sex_label == "Male" else 0,
    "cp": CHEST_PAIN[chest_label],
    "trestbps": trestbps,
    "chol": chol,
    "fbs": int(fbs_high),
    "restecg": ECG[ecg_label],
    "thalach": thalach,
    "exang": int(exang),
    "oldpeak": oldpeak,
    "slope": SLOPE[slope_label],
    "ca": ca,
    "thal": THAL[thal_label],
}
row = pd.DataFrame([patient])[FEATURES]
risk = float(pipeline.predict_proba(row)[0, 1])

# ---------- main page ----------
st.title("CardioScope")
st.write("An educational tool that estimates the chance of heart disease from 13 clinical measurements.")

tab_result, tab_drivers, tab_about = st.tabs(["Result", "What drives the score", "About the model"])

with tab_result:
    if risk < 0.30:
        band, show = "Low", st.success
        advice = "The inputs look similar to patients without heart disease."
    elif risk < 0.60:
        band, show = "Moderate", st.warning
        advice = "The inputs sit in a grey zone. A clinician would want more information."
    else:
        band, show = "High", st.error
        advice = "The inputs resemble patients who were diagnosed with heart disease."

    left, right = st.columns([1, 2])
    left.metric("Estimated risk", f"{risk * 100:.0f}%")
    left.metric("Risk band", band)
    right.progress(risk)
    show(f"**{band} risk.** {advice}")
    right.caption("This is a statistical estimate, not a diagnosis. Talk to a doctor about any health concern.")

    st.subheader("Your inputs")
    summary = pd.DataFrame(
        {"Measurement": [NICE_NAMES[f] for f in FEATURES], "Value": [str(patient[f]) for f in FEATURES]}
    )
    st.dataframe(summary, hide_index=True, use_container_width=True)

with tab_drivers:
    st.subheader("Which measurements matter most to the model")
    imp = pd.Series(bundle["importances"]).rename(index=NICE_NAMES).sort_values(ascending=False)
    st.bar_chart(imp)

    st.subheader("You compared with the training patients")
    cmp_df = pd.DataFrame(
        {
            "You": pd.Series(patient),
            "Avg. healthy patient": pd.Series(bundle["healthy_means"]),
            "Avg. patient with disease": pd.Series(bundle["disease_means"]),
        }
    ).loc[FEATURES]
    cmp_df.index = [NICE_NAMES[f] for f in cmp_df.index]
    st.dataframe(cmp_df, use_container_width=True)

with tab_about:
    st.subheader("About the model")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hold-out accuracy", f"{metrics['holdout_accuracy'] * 100:.1f}%")
    c2.metric("Hold-out ROC AUC", f"{metrics['holdout_auc']:.2f}")
    c3.metric("Training patients", metrics["n_samples"])
    st.markdown(
        """
- **Data:** UCI Cleveland heart disease dataset (303 patients). The 0-4 diagnosis scale was collapsed to *disease / no disease*.
- **Model:** median imputation, standard scaling, then a 300-tree Random Forest.
- **Limits:** small, old, single-hospital data. Do not use it for real medical decisions.
"""
    )
