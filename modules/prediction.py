import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from .data_processing import FEATURES

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "trained_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def predict_one(values: dict):
    model = load_model()
    row = pd.DataFrame([{feature: values[feature] for feature in FEATURES}], columns=FEATURES)
    prediction = int(model.predict(row)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(row)[0]
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(probs[classes.index(1)])
    return prediction, probability
