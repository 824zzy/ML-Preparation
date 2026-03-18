"""
Softmax Regression (Multinomial Logistic Regression) from Scratch using NumPy

Multi-class classification using softmax activation and cross-entropy loss.

Key formulas:
- Softmax: σ(z)_i = e^(z_i) / Σ_j e^(z_j)
- Prediction: ŷ = softmax(W^T X + b)
- Cross-Entropy Loss: L = -(1/n) Σ_i Σ_k y_ik log(ŷ_ik)
  where k is the class index, i is the sample index
- Gradient: dL/dW = (1/n) X^T (ŷ - y)

This is the generalization of binary logistic regression to multiple classes.
"""

import numpy as np


def softmax(z):
    """Numerically stable softmax function."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot_encode(y, n_classes):
    """Convert class labels to one-hot encoded format."""
    n_samples = len(y)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y] = 1
    return one_hot


class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.n_classes = None

    def cross_entropy_loss(self, y_true, y_pred):
        """
        Compute cross-entropy loss.

        Args:
            y_true: One-hot encoded true labels (n_samples, n_classes)
            y_pred: Predicted probabilities (n_samples, n_classes)
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def fit(self, X, y):
        """
        Train softmax regression using gradient descent.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) with values 0 to n_classes-1
        """
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))

        # Initialize parameters
        self.weights = np.zeros((n_features, self.n_classes))
        self.bias = np.zeros((1, self.n_classes))

        # Convert labels to one-hot encoding
        y_one_hot = one_hot_encode(y, self.n_classes)

        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            logits = np.dot(X, self.weights) + self.bias  # (n_samples, n_classes)
            y_pred = softmax(logits)

            # Compute loss
            loss = self.cross_entropy_loss(y_one_hot, y_pred)

            # Compute gradients
            error = y_pred - y_one_hot  # (n_samples, n_classes)
            dw = (1 / n_samples) * np.dot(X.T, error)  # (n_features, n_classes)
            db = (1 / n_samples) * np.sum(error, axis=0, keepdims=True)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Print progress
            if i % 100 == 0:
                accuracy = np.mean(self.predict(X) == y)
                print(f"Iteration {i}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    def predict_proba(self, X):
        """Return probability predictions for each class."""
        logits = np.dot(X, self.weights) + self.bias
        return softmax(logits)

    def predict(self, X):
        """Return class predictions."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


if __name__ == "__main__":
    # Generate synthetic 3-class classification data
    np.random.seed(42)

    # Class 0: centered around (-3, -3)
    X0 = np.random.randn(100, 2) + np.array([-3, -3])
    y0 = np.zeros(100, dtype=int)

    # Class 1: centered around (3, 3)
    X1 = np.random.randn(100, 2) + np.array([3, 3])
    y1 = np.ones(100, dtype=int)

    # Class 2: centered around (0, 4)
    X2 = np.random.randn(100, 2) + np.array([0, 4])
    y2 = np.full(100, 2, dtype=int)

    # Combine data
    X = np.vstack([X0, X1, X2])
    y = np.hstack([y0, y1, y2])

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Train model
    print("Training Softmax Regression (3 classes)...")
    print("=" * 60)
    model = SoftmaxRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    # Final predictions
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)

    print("\n" + "=" * 60)
    print(f"Final Training Accuracy: {accuracy:.4f}")

    # Show weight matrix
    print(f"\nWeight Matrix shape: {model.weights.shape}")
    print(f"Bias shape: {model.bias.shape}")

    # Test on specific points
    test_points = np.array([[-3, -3], [3, 3], [0, 4], [0, 0]])
    test_probs = model.predict_proba(test_points)
    test_preds = model.predict(test_points)

    print("\n" + "=" * 60)
    print("Test Predictions:")
    for i, point in enumerate(test_points):
        print(f"\nPoint {point}:")
        print(f"  Probabilities: {test_probs[i]}")
        print(f"  Predicted class: {test_preds[i]}")

    # Show confusion on training data
    print("\n" + "=" * 60)
    print("Class Distribution:")
    for c in range(model.n_classes):
        true_count = np.sum(y == c)
        pred_count = np.sum(predictions == c)
        correct = np.sum((y == c) & (predictions == c))
        print(
            f"Class {c}: {true_count} samples, {pred_count} predicted, {correct} correct"
        )
