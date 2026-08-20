"""
Step 5: Training Loop (Gradient Descent with Momentum) + Evaluation
Neural Network from Scratch + Fraud Detection
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

np.random.seed(42)


class NeuralNetworkFromScratch:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.params = {}
        self._initialize_weights()

    def _initialize_weights(self):
        for l in range(1, self.num_layers + 1):
            fan_in = self.layer_sizes[l - 1]
            fan_out = self.layer_sizes[l]
            self.params[f"W{l}"] = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            self.params[f"b{l}"] = np.zeros((1, fan_out))

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_deriv(Z):
        return (Z > 0).astype(float)

    @staticmethod
    def sigmoid(Z):
        Z_clipped = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z_clipped))

    def forward(self, X):
        cache = {"A0": X}
        A = X
        for l in range(1, self.num_layers):
            W, b = self.params[f"W{l}"], self.params[f"b{l}"]
            Z = A @ W + b
            A = self.relu(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A
        L = self.num_layers
        W, b = self.params[f"W{L}"], self.params[f"b{L}"]
        Z = A @ W + b
        A = self.sigmoid(Z)
        cache[f"Z{L}"] = Z
        cache[f"A{L}"] = A
        return A, cache

    @staticmethod
    def compute_loss(A_out, Y, w0, w1, eps=1e-8):
        N = Y.shape[0]
        A_clipped = np.clip(A_out, eps, 1 - eps)
        loss = -(1.0 / N) * np.sum(
            w1 * Y * np.log(A_clipped) + w0 * (1 - Y) * np.log(1 - A_clipped)
        )
        return loss

    def backward(self, cache, Y, w0, w1):
        grads = {}
        N = Y.shape[0]
        L = self.num_layers

        A_L = cache[f"A{L}"]
        dZ = w0 * (1 - Y) * A_L - w1 * Y * (1 - A_L)

        for l in range(L, 0, -1):
            A_prev = cache[f"A{l-1}"]
            grads[f"dW{l}"] = (1.0 / N) * (A_prev.T @ dZ)
            grads[f"db{l}"] = (1.0 / N) * np.sum(dZ, axis=0, keepdims=True)

            if l > 1:
                W = self.params[f"W{l}"]
                dA_prev = dZ @ W.T
                Z_prev = cache[f"Z{l-1}"]
                dZ = dA_prev * self.relu_deriv(Z_prev)

        return grads

    def init_momentum(self):
        """Velocity terms for momentum GD, one per parameter, initialized to zero."""
        velocity = {}
        for key, val in self.params.items():
            velocity[key] = np.zeros_like(val)
        return velocity

    def update_params(self, grads, velocity, learning_rate, momentum_beta):
        """
        Momentum update:
            v = beta * v + (1 - beta) * grad
            param = param - learning_rate * v
        beta=0 reduces exactly to vanilla gradient descent.
        """
        for l in range(1, self.num_layers + 1):
            for p in ["W", "b"]:
                key = f"{p}{l}"
                grad_key = f"d{key}"
                velocity[key] = momentum_beta * velocity[key] + (1 - momentum_beta) * grads[grad_key]
                self.params[key] -= learning_rate * velocity[key]

    def predict_proba(self, X):
        A_out, _ = self.forward(X)
        return A_out

    def train(self, X_train, Y_train, w0, w1, epochs=500, learning_rate=0.05,
               momentum_beta=0.9, print_every=50):
        velocity = self.init_momentum()
        loss_history = []

        for epoch in range(1, epochs + 1):
            A_out, cache = self.forward(X_train)
            loss = self.compute_loss(A_out, Y_train, w0, w1)
            loss_history.append(loss)

            grads = self.backward(cache, Y_train, w0, w1)
            self.update_params(grads, velocity, learning_rate, momentum_beta)

            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch:4d}/{epochs} - weighted loss: {loss:.6f}")

        return loss_history


def evaluate(net, X, Y, threshold=0.5, label=""):
    probs = net.predict_proba(X).ravel()
    preds = (probs >= threshold).astype(int)
    Y_flat = Y.ravel()

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
    DATA_DIR = "data"
    X_train = np.load(f"{DATA_DIR}/X_train_scaled.npy")
    X_test = np.load(f"{DATA_DIR}/X_test_scaled.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy").reshape(-1, 1).astype(float)
    y_test = np.load(f"{DATA_DIR}/y_test.npy").reshape(-1, 1).astype(float)

    N = len(y_train)
    n_pos = y_train.sum()
    n_neg = N - n_pos
    w1 = N / (2.0 * n_pos)
    w0 = N / (2.0 * n_neg)

    net = NeuralNetworkFromScratch(layer_sizes=[X_train.shape[1], 16, 8, 1])

    print("Training NumPy network (weighted loss, momentum gradient descent)...")
    loss_history = net.train(
        X_train, y_train, w0, w1,
        epochs=500, learning_rate=0.05, momentum_beta=0.9, print_every=50
    )

    evaluate(net, X_train, y_train, threshold=0.5, label="TRAIN")
    test_metrics = evaluate(net, X_test, y_test, threshold=0.5, label="TEST")

    plt.figure(figsize=(7, 4.5))
    plt.plot(loss_history, color="#1F4E79")
    plt.title("Training Loss (Weighted BCE) - NumPy Network")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE Loss")
    plt.tight_layout()
    plt.savefig("numpy_loss_curve.png")
    print("\nSaved numpy_loss_curve.png")

    np.savez("numpy_model_params.npz", **net.params)
    print("Saved numpy_model_params.npz")