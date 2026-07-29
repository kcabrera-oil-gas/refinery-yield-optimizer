import streamlit as st
import numpy as np
import joblib, os

st.set_page_config(page_title="Refinery Yield Optimizer", layout="wide")
st.title("Refinery Yield Optimizer")
st.markdown("Crude distillation unit yield prediction and product slate optimization using ML")

model_data = joblib.load(os.path.join("outputs", "models", "model.pkl"))
model = model_data["model"]
scaler = model_data["scaler"]
feats = model_data["feature_names"]

cols = st.columns(3)
inputs = {}
for i, f in enumerate(feats):
    with cols[i % 3]:
        inputs[f] = st.number_input(f.replace("_", " ").title(), value=0.0, key=f)

if st.button("Predict"):
    X = np.array([[inputs[f] for f in feats]])
    if scaler:
        X = scaler.transform(X)
    pred = model.predict(X)[0]
    val = float(pred) if hasattr(pred, '__float__') else str(pred)
    st.metric("Prediction", f"{val:.4f}" if isinstance(val, float) else val)
    st.success("Inference complete")
