"""
Step 10: FastAPI Backend
Neural Network from Scratch + Fraud Detection

Serves the final trained model (SMOTE + RMSprop + L2, from Step 9) via:
  POST /predict     - fraud probability + flagged/not-flagged decision
  GET  /model-info  - which config is being served + its test metrics

Run with: uvicorn app:app --reload   (save this file as app.py)
Then open: http://127.0.0.1:8000/docs for interactive API docs.
"""

import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

DATA_DIR = "data"

# ---- Load config, scaler, and model at startup ----
with open("model_config.json") as f:
    CONFIG = json.load(f)

scaler = joblib.load(f"{DATA_DIR}/scaler.joblib")

MERCHANT_CATEGORIES = CONFIG["merchant_categories"]  # ["Clothing", "Electronics", "Food", "Grocery", "Travel"]
FEATURE_ORDER = CONFIG["feature_order"]
THRESHOLD = CONFIG["decision_threshold"]


class FraudNet(nn.Module):
    """Must match the architecture used in Step 9 exactly, or loading weights will fail."""
    def __init__(self, input_dim, dropout_rate=0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, 16), nn.ReLU()]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers += [nn.Linear(16, 8), nn.ReLU()]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers += [nn.Linear(8, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = FraudNet(input_dim=CONFIG["input_dim"], dropout_rate=CONFIG["dropout_rate"])
model.load_state_dict(torch.load("final_fraud_model.pt", map_location="cpu"))
model.eval()


# ---- Request schema ----
class Transaction(BaseModel):
    amount: float = Field(..., ge=0, description="Transaction amount")
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    merchant_category: Literal["Clothing", "Electronics", "Food", "Grocery", "Travel"]
    foreign_transaction: Literal[0, 1]
    location_mismatch: Literal[0, 1]
    device_trust_score: int = Field(..., ge=0, le=100)
    velocity_last_24h: int = Field(..., ge=0)
    cardholder_age: int = Field(..., ge=18, le=120)

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 245.50,
                "transaction_hour": 3,
                "merchant_category": "Electronics",
                "foreign_transaction": 1,
                "location_mismatch": 1,
                "device_trust_score": 28,
                "velocity_last_24h": 6,
                "cardholder_age": 34
            }
        }


def build_feature_vector(txn: Transaction) -> np.ndarray:
    """Builds the 12-column feature vector in the exact order the scaler/model expect."""
    row = {
        "amount": txn.amount,
        "transaction_hour": txn.transaction_hour,
        "foreign_transaction": txn.foreign_transaction,
        "location_mismatch": txn.location_mismatch,
        "device_trust_score": txn.device_trust_score,
        "velocity_last_24h": txn.velocity_last_24h,
        "cardholder_age": txn.cardholder_age,
    }
    # One-hot encode merchant_category into the 5 cat_* columns
    for cat in MERCHANT_CATEGORIES:
        row[f"cat_{cat}"] = 1 if txn.merchant_category == cat else 0

    # Order exactly per FEATURE_ORDER — critical, do not rely on dict insertion order
    ordered = [row[col] for col in FEATURE_ORDER]
    return np.array(ordered, dtype=np.float32).reshape(1, -1)


app = FastAPI(
    title="Fraud Detection API",
    description="Serves a PyTorch neural network (SMOTE + RMSprop + L2) trained from scratch, "
                 "then rebuilt in PyTorch, for real-time transaction fraud scoring.",
    version="1.0.0",
)


@app.get("/model-info")
def model_info():
    return {
        "architecture": f"{CONFIG['input_dim']} -> " + " -> ".join(str(h) for h in CONFIG["hidden_layers"]) + " -> 1",
        "optimizer": CONFIG["optimizer"],
        "imbalance_strategy": CONFIG["imbalance_strategy"],
        "decision_threshold": THRESHOLD,
        "test_metrics": CONFIG["test_metrics"],
    }


@app.post("/predict")
def predict(txn: Transaction):
    try:
        features = build_feature_vector(txn)
        features_scaled = scaler.transform(features)
        X = torch.tensor(features_scaled, dtype=torch.float32)

        with torch.no_grad():
            logit = model(X)
            probability = torch.sigmoid(logit).item()

        flagged = probability >= THRESHOLD

        return {
            "fraud_probability": round(probability, 4),
            "flagged_as_fraud": flagged,
            "decision_threshold": THRESHOLD,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running. See /docs for usage."}