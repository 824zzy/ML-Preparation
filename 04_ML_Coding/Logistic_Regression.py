"""
Logistic Regression from Scratch using NumPy

Binary classification using logistic regression with gradient descent.

Key formulas:
- Sigmoid: σ(z) = 1 / (1 + e^(-z))
- Prediction: ŷ = σ(w^T x + b)
- Binary Cross-Entropy Loss: L = -(1/n) Σ [y log(ŷ) + (1-y) log(1-ŷ)]
- Gradients:
  - dL/dw = (1/n) X^T (ŷ - y)
  - dL/db = (1/n) Σ (ŷ - y)
"""

import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow

    def binary_cross_entropy(self, y_true, y_pred):
        """Binary cross-entropy loss."""
        epsilon = 1e-15  # Prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        """
        Train logistic regression using gradient descent.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) with values 0 or 1
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_output)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Print loss every 100 iterations
            if i % 100 == 0:
                loss = self.binary_cross_entropy(y, y_pred)
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        """Return probability predictions."""
        linear_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)

    def predict(self, X):
        """Return class predictions (0 or 1)."""
        return (self.predict_proba(X) >= 0.5).astype(int)


if __name__ == "__main__":
    # Generate synthetic binary classification data
    np.random.seed(42)

    # Class 0: centered around (-2, -2)
    X0 = np.random.randn(50, 2) + np.array([-2, -2])
    y0 = np.zeros(50)

    # Class 1: centered around (2, 2)
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    y1 = np.ones(50)

    # Combine data
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Train model
    print("Training Logistic Regression...")
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)

    print(f"\nFinal Model Parameters:")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias:.4f}")
    print(f"Training Accuracy: {accuracy:.4f}")

    # Test on a few sample points
    test_points = np.array([[-2, -2], [2, 2], [0, 0]])
    test_probs = model.predict_proba(test_points)
    test_preds = model.predict(test_points)

    print(f"\nTest Predictions:")
    for i, point in enumerate(test_points):
        print(f"Point {point}: P(y=1) = {test_probs[i]:.4f}, Predicted class = {test_preds[i]}")
