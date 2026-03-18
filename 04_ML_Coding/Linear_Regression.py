"""
Linear Regression from Scratch using NumPy

Two approaches: Normal Equation and Gradient Descent

Key formulas:
- Prediction: ŷ = w^T x + b
- MSE Loss: L = (1/2n) Σ (ŷ - y)^2
- Normal Equation: w = (X^T X)^(-1) X^T y
- Gradient Descent:
  - dL/dw = (1/n) X^T (ŷ - y)
  - dL/db = (1/n) Σ (ŷ - y)
"""

import numpy as np


class LinearRegression:
    def __init__(self, method="gradient_descent", learning_rate=0.01, n_iterations=1000):
        """
        Args:
            method: 'gradient_descent' or 'normal_equation'
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations for gradient descent
        """
        self.method = method
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def mse_loss(self, y_true, y_pred):
        """Mean squared error loss."""
        return np.mean((y_pred - y_true) ** 2)

    def fit(self, X, y):
        """
        Train linear regression.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
        """
        n_samples, n_features = X.shape

        if self.method == "normal_equation":
            # Add bias term to X
            X_with_bias = np.c_[np.ones(n_samples), X]

            # Normal equation: θ = (X^T X)^(-1) X^T y
            theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

            self.bias = theta[0]
            self.weights = theta[1:]

            y_pred = self.predict(X)
            loss = self.mse_loss(y, y_pred)
            print(f"Normal Equation - Final Loss: {loss:.4f}")

        elif self.method == "gradient_descent":
            # Initialize parameters
            self.weights = np.zeros(n_features)
            self.bias = 0

            # Gradient descent
            for i in range(self.n_iterations):
                # Forward pass
                y_pred = np.dot(X, self.weights) + self.bias

                # Compute gradients
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
                db = (1 / n_samples) * np.sum(y_pred - y)

                # Update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

                # Print loss every 100 iterations
                if i % 100 == 0:
                    loss = self.mse_loss(y, y_pred)
                    print(f"Iteration {i}: Loss = {loss:.4f}")
        else:
            raise ValueError("Method must be 'gradient_descent' or 'normal_equation'")

    def predict(self, X):
        """Make predictions."""
        return np.dot(X, self.weights) + self.bias


if __name__ == "__main__":
    # Generate synthetic regression data
    # True relationship: y = 3x1 + 2x2 + 5 + noise
    np.random.seed(42)

    n_samples = 100
    X = np.random.randn(n_samples, 2)
    true_weights = np.array([3, 2])
    true_bias = 5
    noise = np.random.randn(n_samples) * 0.5

    y = X @ true_weights + true_bias + noise

    print("=" * 50)
    print("Method 1: Gradient Descent")
    print("=" * 50)
    model_gd = LinearRegression(method="gradient_descent", learning_rate=0.1, n_iterations=1000)
    model_gd.fit(X, y)

    print(f"\nLearned Parameters (Gradient Descent):")
    print(f"Weights: {model_gd.weights}")
    print(f"Bias: {model_gd.bias:.4f}")
    print(f"True Weights: {true_weights}")
    print(f"True Bias: {true_bias}")

    print("\n" + "=" * 50)
    print("Method 2: Normal Equation")
    print("=" * 50)
    model_ne = LinearRegression(method="normal_equation")
    model_ne.fit(X, y)

    print(f"\nLearned Parameters (Normal Equation):")
    print(f"Weights: {model_ne.weights}")
    print(f"Bias: {model_ne.bias:.4f}")

    # Test predictions
    X_test = np.array([[1, 1], [0, 0], [-1, -1]])
    print(f"\nTest Predictions:")
    for i, point in enumerate(X_test):
        pred_gd = model_gd.predict(point.reshape(1, -1))[0]
        pred_ne = model_ne.predict(point.reshape(1, -1))[0]
        true_val = point @ true_weights + true_bias
        print(f"Point {point}: GD={pred_gd:.2f}, NE={pred_ne:.2f}, True={true_val:.2f}")
