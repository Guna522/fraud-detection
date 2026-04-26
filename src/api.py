from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from src.features import add_features

# Initialize app
app = FastAPI()

# Load trained model
model = joblib.load("model.pkl")


# -----------------------------
# Input Schema (Validation)
# -----------------------------
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: Transaction):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])

        # Apply feature engineering
        df = add_features(df)

        # Align columns with training
        expected_cols = model.get_booster().feature_names
        df = df.reindex(columns=expected_cols, fill_value=0)

        # Predict
        proba = model.predict_proba(df)[0][1]
        prediction = "fraud" if proba > 0.3 else "not fraud"

        return {
            "fraud_probability": float(proba),
            "prediction": prediction
        }

    except Exception as e:
        return {"error": str(e)}