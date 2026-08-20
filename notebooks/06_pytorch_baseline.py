"""
Step 6: PyTorch Rebuild (baseline-equivalent to the NumPy network)
Neural Network from Scratch + Fraud Detection

Goal: recreate the SAME architecture and weighted-loss idea as the NumPy
implementation, so its results can be used to sanity-check that the
from-scratch forward/backward pass was implemented correctly.
Optimizer/regularization comparisons happen in Step 7, not here.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "data"


# ---- Same architecture as the NumPy network: input -> 16 -> 8 -> 1 ----
class FraudNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            # no sigmoid here — BCEWithLogitsLoss applies it internally (more stable)
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, X, Y, threshold=0.5, label=""):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits).numpy().ravel()
    preds = (probs >= threshold).astype(int)
    Y_flat = Y.numpy().ravel()

    precision = precision_score(Y_flat, preds, zero_division=0)
    recall = recall_score(Y_flat, preds, zero_division=0)
    f1 = f1_score(Y_flat, preds, zero_division=0)
    pr_auc = average_precision_score(Y_flat, probs)

    print(f"\n--- Evaluation: {label} (threshold={threshold}) ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print("\nConfusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(Y_flat, preds))
    print("\nFull classification report:")
    print(classification_report(Y_flat, preds, target_names=["Legit", "Fraud"], zero_division=0))

    return {"precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc}


if __name__ == "__main__":
    # ---- Load the same preprocessed data used by the NumPy network ----
    X_train_np = np.load(f"{DATA_DIR}/X_train_scaled.npy")
    X_test_np = np.load(f"{DATA_DIR}/X_test_scaled.npy")
    y_train_np = np.load(f"{DATA_DIR}/y_train.npy").reshape(-1, 1).astype(np.float32)
    y_test_np = np.load(f"{DATA_DIR}/y_test.npy").reshape(-1, 1).astype(np.float32)

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)

    # ---- Same class-weighting idea as the NumPy version ----
    # NumPy used two separate weights (w0 for negatives, w1 for positives) inside the loss.
    # BCEWithLogitsLoss's pos_weight achieves the same effect: it multiplies the
    # positive-class term by pos_weight, leaving the negative term at weight 1.
    # To match the NumPy w0/w1 ratio exactly, we use pos_weight = w1 / w0.
    N = len(y_train)
    n_pos = y_train.sum().item()
    n_neg = N - n_pos
    w1 = N / (2.0 * n_pos)
    w0 = N / (2.0 * n_neg)
    pos_weight = torch.tensor([w1 / w0], dtype=torch.float32)
    print(f"n_pos={n_pos}, n_neg={n_neg}, pos_weight={pos_weight.item():.4f}")

    model = FraudNet(input_dim=X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Adam here only as a baseline-equivalent optimizer; Step 7 compares this
    # against SGD and RMSprop deliberately.
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 500
    loss_history = []

    print("\nTraining PyTorch network (baseline-equivalent architecture)...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} - weighted loss: {loss.item():.6f}")

    evaluate(model, X_train, y_train, threshold=0.5, label="TRAIN")
    test_metrics = evaluate(model, X_test, y_test, threshold=0.5, label="TEST")

    plt.figure(figsize=(7, 4.5))
    plt.plot(loss_history, color="#C0392B")
    plt.title("Training Loss (Weighted BCE) - PyTorch Network (Adam)")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE Loss")
    plt.tight_layout()
    plt.savefig("pytorch_baseline_loss_curve.png")
    print("\nSaved pytorch_baseline_loss_curve.png")

    torch.save(model.state_dict(), "pytorch_baseline_model.pt")
    print("Saved pytorch_baseline_model.pt")