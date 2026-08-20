"""
Step 3: Neural Network from Scratch - Forward Pass
Neural Network from Scratch + Fraud Detection

Architecture: input(12) -> hidden(16, ReLU) -> hidden(8, ReLU) -> output(1, Sigmoid)

This module defines the network's parameters and forward computation only.
Loss + backprop are added in Step 4; the training loop is added in Step 5.
"""

import numpy as np

np.random.seed(42)


class NeuralNetworkFromScratch:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list like [12, 16, 8, 1] -> input size, hidden sizes..., output size
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # number of weight matrices
        self.params = {}
        self._initialize_weights()

    def _initialize_weights(self):
        """
        He initialization for ReLU hidden layers: W ~ N(0, sqrt(2 / fan_in)).
        This keeps activation variance stable across layers, which matters a lot —
        plain small-random or zero init causes vanishing gradients or symmetric,
        non-differentiating neurons.
        """
        for l in range(1, self.num_layers + 1):
            fan_in = self.layer_sizes[l - 1]
            fan_out = self.layer_sizes[l]
            self.params[f"W{l}"] = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            self.params[f"b{l}"] = np.zeros((1, fan_out))

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def sigmoid(Z):
        # Clip to avoid overflow in exp() for very negative Z
        Z_clipped = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z_clipped))

    def forward(self, X):
        """
        X: shape (num_samples, num_features)
        Returns: final output A (shape (num_samples, 1)) and a cache dict
                 holding every Z and A per layer, needed for backprop.
        """
        cache = {"A0": X}
        A = X

        # Hidden layers: Linear -> ReLU
        for l in range(1, self.num_layers):
            W, b = self.params[f"W{l}"], self.params[f"b{l}"]
            Z = A @ W + b
            A = self.relu(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        # Output layer: Linear -> Sigmoid
        L = self.num_layers
        W, b = self.params[f"W{L}"], self.params[f"b{L}"]
        Z = A @ W + b
        A = self.sigmoid(Z)
        cache[f"Z{L}"] = Z
        cache[f"A{L}"] = A

        return A, cache


if __name__ == "__main__":
    # Quick sanity check with real data
    X_train = np.load("data/X_train_scaled.npy")
    y_train = np.load("data/y_train.npy")

    print("X_train shape:", X_train.shape)

    net = NeuralNetworkFromScratch(layer_sizes=[X_train.shape[1], 16, 8, 1])

    print("\nInitialized parameter shapes:")
    for key, val in net.params.items():
        print(f"  {key}: {val.shape}")

    # Run a forward pass on a small batch to confirm shapes flow correctly
    batch = X_train[:5]
    output, cache = net.forward(batch)

    print("\nForward pass on first 5 samples:")
    print("Output (fraud probabilities):\n", output.ravel())
    print("True labels:                 ", y_train[:5])

    print("\nCache keys (needed for backprop):", list(cache.keys()))