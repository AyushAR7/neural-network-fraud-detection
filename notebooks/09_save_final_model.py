"""
Step 9: Save the Winning Model
Neural Network from Scratch + Fraud Detection

Final config (chosen from Steps 7-8):
  - Optimizer: RMSprop
  - Regularization: L2 (weight_decay=1e-3), no dropout
  - Imbalance handling: SMOTE oversampling on training data
  - Early stopping on validation loss

Retrains this exact config on train+val combined (more data for the final
model) then saves the model weights and the fitted scaler for deployment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import copy
import json

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "data"
BEST_OPTIMIZER = "rmsprop"
BEST_LR = 0.01
BEST_WEIGHT_DECAY = 1e-3
BEST_DROPOUT = 0.0


class FraudNet(nn.Module):
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


def evaluate(model, X, Y, threshold=0.5, label=""):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).numpy().ravel()
    preds = (probs >= threshold).astype(int)
    Y_flat = Y.numpy().ravel()
    metrics = {
        "precision": precision_score(Y_flat, preds, zero_division=0),
        "recall": recall_score(Y_flat, preds, zero_division=0),
        "f1": f1_score(Y_flat, preds, zero_division=0),
        "pr_auc": average_precision_score(Y_flat, probs),
    }
    print(f"\n--- {label} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(Y_flat, preds))
    return metrics


if __name__ == "__main__":
    X_train_full = np.load(f"{DATA_DIR}/X_train_scaled.npy")
    X_test_np = np.load(f"{DATA_DIR}/X_test_scaled.npy")
    y_train_full = np.load(f"{DATA_DIR}/y_train.npy").reshape(-1, 1).astype(np.float32)
    y_test_np = np.load(f"{DATA_DIR}/y_test.npy").reshape(-1, 1).astype(np.float32)

    # Keep a small validation slice for early stopping, same split logic as Steps 7-8
    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )

    # Apply SMOTE to the training portion only (never val/test)
    smote = SMOTE(random_state=42)
    X_tr_smote, y_tr_smote = smote.fit_resample(X_tr_np, y_tr_np.ravel())
    print(f"After SMOTE: {X_tr_smote.shape}, fraud count: {int(y_tr_smote.sum())}")

    X_tr = torch.tensor(X_tr_smote, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_smote.reshape(-1, 1), dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)

    input_dim = X_tr.shape[1]
    model = FraudNet(input_dim, dropout_rate=BEST_DROPOUT)
    criterion = nn.BCEWithLogitsLoss()  # unweighted - SMOTE already balanced the data
    optimizer = optim.RMSprop(model.parameters(), lr=BEST_LR, weight_decay=BEST_WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0
    patience = 30
    max_epochs = 500

    print("\nTraining final model (SMOTE + RMSprop + L2 + early stopping)...")
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{max_epochs} - train loss: {loss.item():.5f}, val loss: {val_loss:.5f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)

    # ---- Final evaluation on held-out test set ----
    test_metrics = evaluate(model, X_test, y_test, threshold=0.5, label="FINAL MODEL - TEST SET")

    # ---- Persist everything needed for deployment ----
    torch.save(model.state_dict(), "final_fraud_model.pt")
    print("\nSaved final_fraud_model.pt")

    # Save architecture + config so the API can rebuild the exact same model
    config = {
        "input_dim": input_dim,
        "hidden_layers": [16, 8],
        "dropout_rate": BEST_DROPOUT,
        "optimizer": BEST_OPTIMIZER,
        "imbalance_strategy": "SMOTE",
        "decision_threshold": 0.5,
        "test_metrics": test_metrics,
        "feature_order": [
            "amount", "transaction_hour", "foreign_transaction", "location_mismatch",
            "device_trust_score", "velocity_last_24h", "cardholder_age",
            "cat_Clothing", "cat_Electronics", "cat_Food", "cat_Grocery", "cat_Travel"
        ],
        "merchant_categories": ["Clothing", "Electronics", "Food", "Grocery", "Travel"]
        # Confirmed from the actual 02_preprocessing.py run output.
        # The API must one-hot encode merchant_category into these 5 cat_* columns,
        # in this exact position (after cardholder_age), before scaling.
    }
    with open("model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Saved model_config.json")