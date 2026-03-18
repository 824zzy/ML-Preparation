"""
Decision Tree Classifier from Scratch using NumPy

Binary decision tree classifier using Gini impurity for splitting.

Key formulas:
- Gini Impurity: Gini(S) = 1 - Σ p_i^2
  where p_i is the proportion of class i in set S
- Information Gain: IG = Gini(parent) - weighted_avg(Gini(children))

Algorithm:
1. For each feature and threshold, compute information gain
2. Split on the feature/threshold with highest gain
3. Recursively build left and right subtrees
4. Stop when max depth reached or node is pure
"""

import numpy as np
from collections import Counter


class Node:
    """Represents a node in the decision tree."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature index to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Class label if leaf node


class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def gini_impurity(self, y):
        """Calculate Gini impurity for a set of labels."""
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def information_gain(self, y_parent, y_left, y_right):
        """Calculate information gain from a split."""
        n = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)

        gini_parent = self.gini_impurity(y_parent)
        gini_left = self.gini_impurity(y_left)
        gini_right = self.gini_impurity(y_right)

        weighted_avg = (n_left / n) * gini_left + (n_right / n) * gini_right
        return gini_parent - weighted_avg

    def best_split(self, X, y):
        """Find the best feature and threshold to split on."""
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                y_left = y[left_mask]
                y_right = y[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Calculate information gain
                gain = self.information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or n_classes == 1
        ):
            # Leaf node: return most common class
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)

        # Find best split
        best_feature, best_threshold, best_gain = self.best_split(X, y)

        if best_feature is None:
            # Cannot split further
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Recursively build left and right subtrees
        left_child = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(best_feature, best_threshold, left_child, right_child)

    def fit(self, X, y):
        """Build the decision tree."""
        self.root = self.build_tree(X, y)

    def _predict_sample(self, x, node):
        """Predict class for a single sample."""
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        """Predict classes for multiple samples."""
        return np.array([self._predict_sample(x, self.root) for x in X])


if __name__ == "__main__":
    # Generate synthetic classification data
    np.random.seed(42)

    # Class 0: x1 < 0 and x2 < 0
    X0 = np.random.randn(50, 2) - 1
    y0 = np.zeros(50, dtype=int)

    # Class 1: x1 > 0 or x2 > 0
    X1 = np.random.randn(50, 2) + 1
    y1 = np.ones(50, dtype=int)

    # Combine data
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Train decision tree
    print("Training Decision Tree Classifier...")
    tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
    tree.fit(X, y)

    # Make predictions
    predictions = tree.predict(X)
    accuracy = np.mean(predictions == y)

    print(f"Training Accuracy: {accuracy:.4f}")

    # Test on new points
    X_test = np.array([[-2, -2], [2, 2], [0, 0], [-1, 1]])
    y_pred = tree.predict(X_test)

    print(f"\nTest Predictions:")
    for i, point in enumerate(X_test):
        print(f"Point {point}: Predicted class = {y_pred[i]}")
