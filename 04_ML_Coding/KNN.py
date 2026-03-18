"""
K-Nearest Neighbors (KNN) Classifier from Scratch using NumPy

Instance-based learning algorithm that classifies based on majority vote
of K nearest neighbors in the training set.

Algorithm:
1. Store training data
2. For each test point:
   a. Calculate distance to all training points
   b. Find K nearest neighbors
   c. Return majority class among K neighbors

Distance metric: Euclidean distance
"""

import numpy as np
from collections import Counter


class KNearestNeighbors:
    def __init__(self, k=3):
        """
        Args:
            k: Number of nearest neighbors to consider
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance between two vectors."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X, y):
        """
        Store training data (KNN is a lazy learner).

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict labels for test data.

        Args:
            X: Test features (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        """Predict label for a single test sample."""
        # Calculate distances to all training samples
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[: self.k]

        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]

        # Return most common label
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common


if __name__ == "__main__":
    # Generate synthetic classification data
    np.random.seed(42)

    # Class 0: centered around (-2, -2)
    X0 = np.random.randn(50, 2) + np.array([-2, -2])
    y0 = np.zeros(50, dtype=int)

    # Class 1: centered around (2, 2)
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    y1 = np.ones(50, dtype=int)

    # Class 2: centered around (0, 3)
    X2 = np.random.randn(50, 2) + np.array([0, 3])
    y2 = np.full(50, 2, dtype=int)

    # Combine data
    X = np.vstack([X0, X1, X2])
    y = np.hstack([y0, y1, y2])

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Split into train and test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train KNN (k=3)
    print("Training K-Nearest Neighbors (k=3)...")
    knn = KNearestNeighbors(k=3)
    knn.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"Test Accuracy: {accuracy:.4f}")

    # Test different values of k
    print(f"\nAccuracy for different values of k:")
    for k in [1, 3, 5, 7, 10]:
        knn_k = KNearestNeighbors(k=k)
        knn_k.fit(X_train, y_train)
        y_pred_k = knn_k.predict(X_test)
        acc_k = np.mean(y_pred_k == y_test)
        print(f"k={k}: Accuracy = {acc_k:.4f}")

    # Test on specific points
    X_custom = np.array([[-2, -2], [2, 2], [0, 3], [0, 0]])
    y_custom_pred = knn.predict(X_custom)

    print(f"\nCustom Test Predictions (k=3):")
    for i, point in enumerate(X_custom):
        print(f"Point {point}: Predicted class = {y_custom_pred[i]}")
