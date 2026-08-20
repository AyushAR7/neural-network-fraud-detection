"""
Step 8: Imbalance-Handling Comparison
Neural Network from Scratch + Fraud Detection

Compares three ways of handling the 1.51% fraud class imbalance, all using
the winning setup from Step 7 (RMSprop optimizer, L2 regularization,
early stopping) so the ONLY variable changing is the imbalance strategy:

  1) Weighted loss   - original imbalance kept, BCEWithLogitsLoss(pos_weight=...)
  2) SMOTE            - synthetic oversampling of minority class, unweighted loss
  3) Undersampling     - majority class trimmed to match minority size, unweighted loss

Validation and test sets are NEVER resampled - only the training set is
resampled for arms 2 and 3, so evaluation always reflects real-world class
balance.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import copy

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "data"

# ---- Winning config from Step 7 ----
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


def get_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def evaluate_quiet(model, X, Y, threshold=0.5):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).numpy().ravel()
    preds = (probs >= threshold).astype(int)
    Y_flat = Y.numpy().ravel()
    return {
        "precision": precision_score(Y_flat, preds, zero_division=0),
        "recall": recall_score(Y_flat, preds, zero_division=0),
        "f1": f1_score(Y_flat, preds, zero_division=0),
        "pr_auc": average_precision_score(Y_flat, probs),
    }


def train_model(X_train, y_train, X_val, y_val, input_dim, pos_weight,
                 optimizer_name, lr, weight_decay, dropout_rate,
                 max_epochs=500, patience=30, verbose_label=""):
    model = FraudNet(input_dim, dropout_rate=dropout_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr, weight_decay)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"[{verbose_label}] Early stopping at epoch {epoch} "
                  f"(best epoch {best_epoch}, best val loss {best_val_loss:.5f})")
            break
    else:
        print(f"[{verbose_label}] Completed all {max_epochs} epochs "
              f"(best epoch {best_epoch}, best val loss {best_val_loss:.5f})")

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_epoch


if __name__ == "__main__":
    X_train_full = np.load(f"{DATA_DIR}/X_train_scaled.npy")
    X_test_np = np.load(f"{DATA_DIR}/X_test_scaled.npy")
    y_train_full = np.load(f"{DATA_DIR}/y_train.npy").reshape(-1, 1).astype(np.float32)
    y_test_np = np.load(f"{DATA_DIR}/y_test.npy").reshape(-1, 1).astype(np.float32)

    # Same train/val split as Step 7 (same random_state -> same split)
    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )

    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)

    print(f"Train (pre-resample): {X_tr_np.shape}, fraud count: {int(y_tr_np.sum())}")
    print(f"Val: {X_val_np.shape}, fraud count: {int(y_val_np.sum())}")
    print(f"Test: {X_test_np.shape}, fraud count: {int(y_test_np.sum())}\n")

    results_summary = []

    # ===== Arm 1: Weighted loss (original imbalance, reweighted loss) =====
    print("=" * 70)
    print("ARM 1: Weighted loss (original class balance)")
    print("=" * 70)
    X_tr_w = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr_w = torch.tensor(y_tr_np, dtype=torch.float32)

    N = len(y_tr_w)
    n_pos = y_tr_w.sum().item()
    n_neg = N - n_pos
    w1 = N / (2.0 * n_pos)
    w0 = N / (2.0 * n_neg)
    pos_weight_arm1 = torch.tensor([w1 / w0], dtype=torch.float32)

    model_w, _, _, _ = train_model(
        X_tr_w, y_tr_w, X_val, y_val, input_dim=X_tr_w.shape[1], pos_weight=pos_weight_arm1,
        optimizer_name=BEST_OPTIMIZER, lr=BEST_LR, weight_decay=BEST_WEIGHT_DECAY,
        dropout_rate=BEST_DROPOUT, verbose_label="weighted-loss"
    )
    test_metrics_w = evaluate_quiet(model_w, X_test, y_test)
    results_summary.append({"strategy": "Weighted loss", **test_metrics_w})
    print(f"  -> Test: P={test_metrics_w['precision']:.3f} R={test_metrics_w['recall']:.3f} "
          f"F1={test_metrics_w['f1']:.3f} PR-AUC={test_metrics_w['pr_auc']:.3f}\n")

    # ===== Arm 2: SMOTE (oversample minority in TRAINING data only) =====
    print("=" * 70)
    print("ARM 2: SMOTE oversampling (unweighted loss)")
    print("=" * 70)
    smote = SMOTE(random_state=42)
    X_tr_smote, y_tr_smote = smote.fit_resample(X_tr_np, y_tr_np.ravel())
    print(f"After SMOTE: {X_tr_smote.shape}, fraud count: {int(y_tr_smote.sum())} "
          f"(balanced to {y_tr_smote.mean()*100:.1f}%)")

    X_tr_s = torch.tensor(X_tr_smote, dtype=torch.float32)
    y_tr_s = torch.tensor(y_tr_smote.reshape(-1, 1), dtype=torch.float32)
    no_weight = torch.tensor([1.0], dtype=torch.float32)

    model_s, _, _, _ = train_model(
        X_tr_s, y_tr_s, X_val, y_val, input_dim=X_tr_s.shape[1], pos_weight=no_weight,
        optimizer_name=BEST_OPTIMIZER, lr=BEST_LR, weight_decay=BEST_WEIGHT_DECAY,
        dropout_rate=BEST_DROPOUT, verbose_label="smote"
    )
    test_metrics_s = evaluate_quiet(model_s, X_test, y_test)
    results_summary.append({"strategy": "SMOTE", **test_metrics_s})
    print(f"  -> Test: P={test_metrics_s['precision']:.3f} R={test_metrics_s['recall']:.3f} "
          f"F1={test_metrics_s['f1']:.3f} PR-AUC={test_metrics_s['pr_auc']:.3f}\n")

    # ===== Arm 3: Random undersampling (trim majority in TRAINING data only) =====
    print("=" * 70)
    print("ARM 3: Random undersampling (unweighted loss)")
    print("=" * 70)
    rus = RandomUnderSampler(random_state=42)
    X_tr_under, y_tr_under = rus.fit_resample(X_tr_np, y_tr_np.ravel())
    print(f"After undersampling: {X_tr_under.shape}, fraud count: {int(y_tr_under.sum())} "
          f"(balanced to {y_tr_under.mean()*100:.1f}%)")

    X_tr_u = torch.tensor(X_tr_under, dtype=torch.float32)
    y_tr_u = torch.tensor(y_tr_under.reshape(-1, 1), dtype=torch.float32)

    model_u, _, _, _ = train_model(
        X_tr_u, y_tr_u, X_val, y_val, input_dim=X_tr_u.shape[1], pos_weight=no_weight,
        optimizer_name=BEST_OPTIMIZER, lr=BEST_LR, weight_decay=BEST_WEIGHT_DECAY,
        dropout_rate=BEST_DROPOUT, verbose_label="undersample"
    )
    test_metrics_u = evaluate_quiet(model_u, X_test, y_test)
    results_summary.append({"strategy": "Undersampling", **test_metrics_u})
    print(f"  -> Test: P={test_metrics_u['precision']:.3f} R={test_metrics_u['recall']:.3f} "
          f"F1={test_metrics_u['f1']:.3f} PR-AUC={test_metrics_u['pr_auc']:.3f}\n")

    # ===== Summary =====
    print("=" * 70)
    print("IMBALANCE-HANDLING STRATEGY SUMMARY (test set)")
    print("=" * 70)
    print(f"{'Strategy':<18}{'Precision':>10}{'Recall':>10}{'F1':>10}{'PR-AUC':>10}")
    for row in results_summary:
        print(f"{row['strategy']:<18}{row['precision']:>10.3f}{row['recall']:>10.3f}"
              f"{row['f1']:>10.3f}{row['pr_auc']:>10.3f}")

    labels = [r["strategy"] for r in results_summary]
    metrics_to_plot = ["precision", "recall", "f1", "pr_auc"]
    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(9, 5.5))
    for i, m in enumerate(metrics_to_plot):
        values = [r[m] for r in results_summary]
        plt.bar(x + i * width, values, width, label=m)
    plt.xticks(x + width * 1.5, labels)
    plt.ylabel("Score")
    plt.title("Imbalance-Handling Strategy Comparison (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("imbalance_strategy_comparison.png")
    print("\nSaved imbalance_strategy_comparison.png")