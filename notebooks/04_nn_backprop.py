"""
Step 4: Weighted Loss + Backpropagation (+ Gradient Check)
Neural Network from Scratch + Fraud Detection
"""

import numpy as np

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

    # ---------------- Loss ----------------
    @staticmethod
    def compute_loss(A_out, Y, w0, w1, eps=1e-8):
        """
        Weighted binary cross-entropy.
        A_out, Y: shape (N, 1)
        w0: weight for the majority (legit) class
        w1: weight for the minority (fraud) class
        """
        N = Y.shape[0]
        A_clipped = np.clip(A_out, eps, 1 - eps)
        loss = -(1.0 / N) * np.sum(
            w1 * Y * np.log(A_clipped) + w0 * (1 - Y) * np.log(1 - A_clipped)
        )
        return loss

    # ---------------- Backward ----------------
    def backward(self, cache, Y, w0, w1):
        """
        Returns a dict of gradients {dW1, db1, dW2, db2, ..., dWL, dbL}.

        Output-layer gradient derivation (sigmoid + weighted BCE combined):
            dZ_L = w0*(1-Y)*A_L - w1*Y*(1-A_L)
        which collapses to the familiar (A - Y) when w0 = w1 = 1.
        """
        grads = {}
        N = Y.shape[0]
        L = self.num_layers

        A_L = cache[f"A{L}"]
        dZ = w0 * (1 - Y) * A_L - w1 * Y * (1 - A_L)  # (N, 1)

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


def numerical_gradient_check(net, X, Y, w0, w1, param_name="W3", num_checks=5, epsilon=1e-5):
    """
    Verifies the analytical backprop gradients against finite-difference
    approximations for a handful of entries in one parameter matrix.
    A tiny relative difference (~1e-6 to 1e-8) confirms backprop is correct.
    """
    A_out, cache = net.forward(X)
    grads = net.backward(cache, Y, w0, w1)
    analytical = grads[f"d{param_name}"]

    param = net.params[param_name]
    flat_shape = param.shape
    rng = np.random.RandomState(0)
    idx_i = rng.randint(0, flat_shape[0], size=num_checks)
    idx_j = rng.randint(0, flat_shape[1], size=num_checks)

    print(f"\nNumerical gradient check on {param_name} ({num_checks} random entries):")
    print(f"{'entry':<12}{'analytical':>15}{'numerical':>15}{'rel. diff':>15}")

    for i, j in zip(idx_i, idx_j):
        original_value = param[i, j]

        param[i, j] = original_value + epsilon
        A_plus, _ = net.forward(X)
        loss_plus = net.compute_loss(A_plus, Y, w0, w1)

        param[i, j] = original_value - epsilon
        A_minus, _ = net.forward(X)
        loss_minus = net.compute_loss(A_minus, Y, w0, w1)

        param[i, j] = original_value  # restore

        numerical = (loss_plus - loss_minus) / (2 * epsilon)
        rel_diff = abs(analytical[i, j] - numerical) / (abs(analytical[i, j]) + abs(numerical) + 1e-8)

        print(f"({i},{j})    {analytical[i, j]:>15.8f}{numerical:>15.8f}{rel_diff:>15.2e}")


if __name__ == "__main__":
    X_train = np.load("data/X_train_scaled.npy")
    y_train = np.load("data/y_train.npy").reshape(-1, 1).astype(float)

    N = len(y_train)
    n_pos = y_train.sum()
    n_neg = N - n_pos
    w1 = N / (2.0 * n_pos)
    w0 = N / (2.0 * n_neg)
    print(f"N={N}, fraud cases={int(n_pos)}, legit cases={int(n_neg)}")
    print(f"Class weights -> w0 (legit)={w0:.4f}, w1 (fraud)={w1:.4f}")

    net = NeuralNetworkFromScratch(layer_sizes=[X_train.shape[1], 16, 8, 1])

    X_batch = X_train[:32]
    Y_batch = y_train[:32]

    A_out, cache = net.forward(X_batch)
    loss = net.compute_loss(A_out, Y_batch, w0, w1)
    print(f"\nInitial weighted loss on batch of 32: {loss:.6f}")

    grads = net.backward(cache, Y_batch, w0, w1)
    print("\nGradient shapes:")
    for key, val in grads.items():
        print(f"  {key}: {val.shape}")

    numerical_gradient_check(net, X_batch, Y_batch, w0, w1, param_name="W3")
    numerical_gradient_check(net, X_batch, Y_batch, w0, w1, param_name="W1")