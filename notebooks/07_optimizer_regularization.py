"""
Step 7: Optimizer + Regularization Experiments (with Early Stopping)
Neural Network from Scratch + Fraud Detection

Two experiment sets:
  A) Optimizer comparison  - SGD vs Adam vs RMSprop (same regularization config)
  B) Regularization comparison - none / dropout / L2 / dropout+L2 (using the
     best optimizer found in A)

Both use early stopping on a held-out validation split carved out of the
training set, monitored on weighted validation loss. The test set is only
touched for reporting per config here to make comparisons visible - in a
stricter setup you'd only evaluate the single final winning config on test.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import matplotlib.pyplot as plt
import copy

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "data"


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
                 optimizer_name="adam", lr=0.01, weight_decay=0.0, dropout_rate=0.0,
                 max_epochs=500, patience=30, verbose_label=""):
    """Trains with early stopping on validation weighted BCE loss.
    Returns the best model (lowest val loss), plus train/val loss history."""
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

    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )

    X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)

    N = len(y_tr)
    n_pos = y_tr.sum().item()
    n_neg = N - n_pos
    w1 = N / (2.0 * n_pos)
    w0 = N / (2.0 * n_neg)
    pos_weight = torch.tensor([w1 / w0], dtype=torch.float32)
    print(f"Train split: {X_tr.shape}, Val split: {X_val.shape}, pos_weight={pos_weight.item():.4f}\n")

    results = []

    # ===== Experiment A: Optimizer comparison =====
    print("=" * 70)
    print("EXPERIMENT A: Optimizer comparison (SGD vs Adam vs RMSprop)")
    print("=" * 70)

    optimizer_configs = [
        {"name": "SGD",     "lr": 0.1,  "weight_decay": 1e-4, "dropout_rate": 0.2},
        {"name": "Adam",    "lr": 0.01, "weight_decay": 1e-4, "dropout_rate": 0.2},
        {"name": "RMSprop", "lr": 0.01, "weight_decay": 1e-4, "dropout_rate": 0.2},
    ]

    optimizer_histories = {}
    for cfg in optimizer_configs:
        label = f"optimizer={cfg['name']}"
        model, train_hist, val_hist, best_epoch = train_model(
            X_tr, y_tr, X_val, y_val, input_dim=X_tr.shape[1], pos_weight=pos_weight,
            optimizer_name=cfg["name"], lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            dropout_rate=cfg["dropout_rate"], max_epochs=500, patience=30, verbose_label=label
        )
        val_metrics = evaluate_quiet(model, X_val, y_val)
        test_metrics = evaluate_quiet(model, X_test, y_test)
        optimizer_histories[cfg["name"]] = {"train": train_hist, "val": val_hist, "best_epoch": best_epoch}
        results.append({"experiment": "optimizer", "config": cfg["name"],
                         **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"  -> Val:  P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f} "
              f"F1={val_metrics['f1']:.3f} PR-AUC={val_metrics['pr_auc']:.3f}")
        print(f"  -> Test: P={test_metrics['precision']:.3f} R={test_metrics['recall']:.3f} "
              f"F1={test_metrics['f1']:.3f} PR-AUC={test_metrics['pr_auc']:.3f}\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (name, hist) in zip(axes, optimizer_histories.items()):
        ax.plot(hist["train"], label="train", color="#1F4E79")
        ax.plot(hist["val"], label="val", color="#C0392B")
        ax.axvline(hist["best_epoch"], color="gray", linestyle="--", alpha=0.6, label="best epoch")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.legend()
    axes[0].set_ylabel("Weighted BCE Loss")
    plt.suptitle("Optimizer Comparison: Train vs Val Loss")
    plt.tight_layout()
    plt.savefig("optimizer_comparison.png")
    print("Saved optimizer_comparison.png\n")

    best_optimizer_row = max([r for r in results if r["experiment"] == "optimizer"],
                              key=lambda r: r["val_pr_auc"])
    best_optimizer_name = best_optimizer_row["config"]
    best_optimizer_cfg = next(c for c in optimizer_configs if c["name"] == best_optimizer_name)
    print(f"Best optimizer by validation PR-AUC: {best_optimizer_name}\n")

    # ===== Experiment B: Regularization comparison =====
    print("=" * 70)
    print(f"EXPERIMENT B: Regularization comparison (using optimizer={best_optimizer_name})")
    print("=" * 70)

    reg_configs = [
        {"label": "none",         "weight_decay": 0.0,  "dropout_rate": 0.0},
        {"label": "dropout only", "weight_decay": 0.0,  "dropout_rate": 0.3},
        {"label": "L2 only",      "weight_decay": 1e-3, "dropout_rate": 0.0},
        {"label": "dropout + L2", "weight_decay": 1e-3, "dropout_rate": 0.3},
    ]

    reg_summary = []
    for cfg in reg_configs:
        label = f"reg={cfg['label']}"
        model, train_hist, val_hist, best_epoch = train_model(
            X_tr, y_tr, X_val, y_val, input_dim=X_tr.shape[1], pos_weight=pos_weight,
            optimizer_name=best_optimizer_name, lr=best_optimizer_cfg["lr"],
            weight_decay=cfg["weight_decay"], dropout_rate=cfg["dropout_rate"],
            max_epochs=500, patience=30, verbose_label=label
        )
        train_metrics = evaluate_quiet(model, X_tr, y_tr)
        test_metrics = evaluate_quiet(model, X_test, y_test)
        gap = train_metrics["recall"] - test_metrics["recall"]
        reg_summary.append({
            "config": cfg["label"], "train_recall": train_metrics["recall"],
            "test_precision": test_metrics["precision"], "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"], "test_pr_auc": test_metrics["pr_auc"],
            "recall_gap": gap,
        })
        print(f"  -> Train: P={train_metrics['precision']:.3f} R={train_metrics['recall']:.3f}")
        print(f"  -> Test:  P={test_metrics['precision']:.3f} R={test_metrics['recall']:.3f} "
              f"F1={test_metrics['f1']:.3f} PR-AUC={test_metrics['pr_auc']:.3f}")
        print(f"  -> Recall gap (train - test): {gap:.3f}\n")

    print("=" * 70)
    print("REGULARIZATION SUMMARY (overfitting check via recall gap)")
    print("=" * 70)
    print(f"{'Config':<16}{'TrainR':>8}{'TestR':>8}{'TestP':>8}{'TestF1':>8}{'PR-AUC':>8}{'Gap':>8}")
    for row in reg_summary:
        print(f"{row['config']:<16}{row['train_recall']:>8.3f}{row['test_recall']:>8.3f}"
              f"{row['test_precision']:>8.3f}{row['test_f1']:>8.3f}{row['test_pr_auc']:>8.3f}"
              f"{row['recall_gap']:>8.3f}")

    plt.figure(figsize=(7, 4.5))
    labels = [r["config"] for r in reg_summary]
    gaps = [r["recall_gap"] for r in reg_summary]
    plt.bar(labels, gaps, color="#1F4E79")
    plt.ylabel("Recall Gap (Train - Test)")
    plt.title("Overfitting by Regularization Config (lower is better)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("regularization_overfitting_gap.png")
    print("\nSaved regularization_overfitting_gap.png")