"""
Neural Network Training Loop from Scratch using NumPy

Implements a 2-layer MLP with backpropagation and SGD optimizer.

Network architecture:
- Input layer -> Hidden layer (ReLU) -> Output layer (Sigmoid)

Key components:
- Forward pass: compute predictions
- Backward pass: compute gradients via backpropagation
- SGD optimizer: update weights using gradients
- Binary cross-entropy loss

Backpropagation formulas:
- dL/dW2 = dL/dz2 @ a1.T
- dL/db2 = dL/dz2
- dL/dW1 = dL/dz1 @ X.T
- dL/db1 = dL/dz1

where dL/dz is the gradient with respect to pre-activation
"""

import numpy as np


def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(a):
    """Derivative of sigmoid function."""
    return a * (1 - a)


def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of ReLU function."""
    return (z > 0).astype(float)


def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class TwoLayerNN:
    """Two-layer neural network with backpropagation."""

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output units
            learning_rate: Learning rate for SGD
        """
        self.lr = learning_rate

        # Initialize weights with small random values
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.zeros((output_size, 1))

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X: Input data (input_size, batch_size)

        Returns:
            y_pred: Predictions (output_size, batch_size)
            cache: Dictionary containing intermediate values
        """
        # Layer 1: Linear -> ReLU
        z1 = np.dot(self.W1, X) + self.b1
        a1 = relu(z1)

        # Layer 2: Linear -> Sigmoid
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = sigmoid(z2)

        cache = {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "X": X}
        return a2, cache

    def backward(self, y_true, cache):
        """
        Backward pass (backpropagation).

        Args:
            y_true: True labels (output_size, batch_size)
            cache: Dictionary with intermediate values from forward pass

        Returns:
            gradients: Dictionary containing all gradients
        """
        m = y_true.shape[1]  # batch size
        X = cache["X"]
        a1 = cache["a1"]
        a2 = cache["a2"]
        z1 = cache["z1"]

        # Gradient of loss w.r.t. output layer (before sigmoid)
        dz2 = a2 - y_true  # Derivative of BCE + sigmoid

        # Gradients for layer 2
        dW2 = (1 / m) * np.dot(dz2, a1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        # Gradient w.r.t. hidden layer activation
        da1 = np.dot(self.W2.T, dz2)

        # Gradient w.r.t. hidden layer (before ReLU)
        dz1 = da1 * relu_derivative(z1)

        # Gradients for layer 1
        dW1 = (1 / m) * np.dot(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_parameters(self, gradients):
        """Update parameters using SGD."""
        self.W1 -= self.lr * gradients["dW1"]
        self.b1 -= self.lr * gradients["db1"]
        self.W2 -= self.lr * gradients["dW2"]
        self.b2 -= self.lr * gradients["db2"]

    def train(self, X, y, epochs=1000, print_every=100):
        """
        Training loop.

        Args:
            X: Training data (input_size, n_samples)
            y: Training labels (output_size, n_samples)
            epochs: Number of training epochs
            print_every: Print loss every N epochs
        """
        for epoch in range(epochs):
            # Forward pass
            y_pred, cache = self.forward(X)

            # Compute loss
            loss = binary_cross_entropy(y, y_pred)

            # Backward pass
            gradients = self.backward(y, cache)

            # Update parameters
            self.update_parameters(gradients)

            # Print progress
            if epoch % print_every == 0:
                accuracy = np.mean((y_pred > 0.5) == y)
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    def predict(self, X):
        """Make predictions."""
        y_pred, _ = self.forward(X)
        return (y_pred > 0.5).astype(int)


if __name__ == "__main__":
    # XOR problem (classic non-linearly separable problem)
    print("Training Neural Network on XOR Problem")
    print("=" * 60)

    # XOR dataset
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # (2, 4)
    y = np.array([[0, 1, 1, 0]])  # (1, 4)

    print("Dataset:")
    for i in range(X.shape[1]):
        print(f"Input: {X[:, i]}, Output: {y[:, i]}")

    # Create and train network
    print("\nTraining...")
    np.random.seed(42)
    nn = TwoLayerNN(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
    nn.train(X, y, epochs=2000, print_every=200)

    # Test predictions
    print("\n" + "=" * 60)
    print("Final Predictions:")
    predictions = nn.predict(X)

    for i in range(X.shape[1]):
        print(
            f"Input: {X[:, i]}, True: {y[:, i]}, Predicted: {predictions[:, i]}"
        )

    # Test learned representation
    print("\n" + "=" * 60)
    print("Hidden Layer Activations:")
    _, cache = nn.forward(X)
    hidden_activations = cache["a1"]

    for i in range(X.shape[1]):
        print(f"Input {X[:, i]}: {hidden_activations[:, i]}")
