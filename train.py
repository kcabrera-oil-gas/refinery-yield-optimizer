import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from data_generator import generate_dataset
import os

MODEL_DIR = "outputs/models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    df = generate_dataset()
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    r2 = r2_score(y_test, preds)
    print(f"R2 score: {r2:.4f}")
    joblib.dump({
        "model": model, "scaler": scaler, "label_encoder": None,
        "feature_names": list(X.columns), "target_name": "target",
        "metrics": {"r2": float(r2)}
    }, os.path.join(MODEL_DIR, "model.pkl"))
    print(f"Model saved to {MODEL_DIR}/model.pkl")

if __name__ == "__main__":
    train()
