"""
Step 2: Preprocessing
Neural Network from Scratch + Fraud Detection

- One-hot encode merchant_category
- Standard-scale numeric features (fit on train only, to avoid leakage)
- Save processed arrays + the fitted scaler/encoder for later use (including deployment)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

DATA_DIR = "data"

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()

print("Before encoding:")
print("X_train columns:", list(X_train.columns))
print("X_train shape:", X_train.shape)

# ---- One-hot encode merchant_category ----
X_train_enc = pd.get_dummies(X_train, columns=["merchant_category"], prefix="cat")
X_test_enc = pd.get_dummies(X_test, columns=["merchant_category"], prefix="cat")

# Align test columns to train columns (in case a category is missing in test split)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

print("\nAfter one-hot encoding:")
print("X_train_enc columns:", list(X_train_enc.columns))
print("X_train_enc shape:", X_train_enc.shape)
print("X_test_enc shape:", X_test_enc.shape)

# ---- Scale all features ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_test_scaled = scaler.transform(X_test_enc)

print("\nScaled feature means (train, should be ~0):", np.round(X_train_scaled.mean(axis=0), 3))
print("Scaled feature stds  (train, should be ~1):", np.round(X_train_scaled.std(axis=0), 3))

# ---- Save everything needed downstream ----
np.save(f"{DATA_DIR}/X_train_scaled.npy", X_train_scaled)
np.save(f"{DATA_DIR}/X_test_scaled.npy", X_test_scaled)
np.save(f"{DATA_DIR}/y_train.npy", y_train)
np.save(f"{DATA_DIR}/y_test.npy", y_test)

joblib.dump(scaler, f"{DATA_DIR}/scaler.joblib")
joblib.dump(list(X_train_enc.columns), f"{DATA_DIR}/feature_columns.joblib")

print("\nSaved: X_train_scaled.npy, X_test_scaled.npy, y_train.npy, y_test.npy")
print("Saved: scaler.joblib, feature_columns.joblib (needed later for the FastAPI deployment)")

print("\nFinal feature order (the API must build inputs in this exact order later):")
for i, col in enumerate(X_train_enc.columns):
    print(f"  {i}: {col}")