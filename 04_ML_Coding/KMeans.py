"""
K-Means Clustering from Scratch using NumPy

Unsupervised clustering algorithm that partitions data into K clusters.

Algorithm:
1. Initialize K cluster centroids randomly (from data points)
2. Repeat until convergence:
   a. Assignment step: Assign each point to nearest centroid
   b. Update step: Recompute centroids as mean of assigned points
3. Convergence: when centroids don't change or max iterations reached

Distance metric: Euclidean distance
"""

import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        """
        Args:
            n_clusters: Number of clusters (K)
            max_iters: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance between two arrays."""
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def fit(self, X):
        """
        Fit K-means clustering.

        Args:
            X: Training data (n_samples, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Initialize centroids randomly from data points
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for iteration in range(self.max_iters):
            # Assignment step: assign each point to nearest centroid
            labels = self._assign_clusters(X)

            # Store old centroids to check convergence
            old_centroids = self.centroids.copy()

            # Update step: recompute centroids
            self.centroids = self._update_centroids(X, labels)

            # Check for convergence
            if np.allclose(old_centroids, self.centroids):
                print(f"Converged at iteration {iteration}")
                break

            if iteration % 10 == 0:
                inertia = self._compute_inertia(X, labels)
                print(f"Iteration {iteration}: Inertia = {inertia:.4f}")

        # Final assignment
        self.labels_ = self._assign_clusters(X)
        final_inertia = self._compute_inertia(X, self.labels_)
        print(f"Final Inertia: {final_inertia:.4f}")

    def _assign_clusters(self, X):
        """Assign each sample to the nearest centroid."""
        distances = np.zeros((X.shape[0], self.n_clusters))

        for k in range(self.n_clusters):
            distances[:, k] = self.euclidean_distance(X, self.centroids[k])

        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """Compute new centroids as mean of points in each cluster."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # Keep old centroid if cluster is empty
                new_centroids[k] = self.centroids[k]

        return new_centroids

    def _compute_inertia(self, X, labels):
        """Compute inertia (sum of squared distances to nearest centroid)."""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                distances = self.euclidean_distance(cluster_points, self.centroids[k])
                inertia += np.sum(distances ** 2)
        return inertia

    def predict(self, X):
        """Predict cluster labels for new data."""
        return self._assign_clusters(X)


if __name__ == "__main__":
    # Generate synthetic clustered data
    np.random.seed(42)

    # Three distinct clusters
    cluster1 = np.random.randn(50, 2) + np.array([0, 0])
    cluster2 = np.random.randn(50, 2) + np.array([5, 5])
    cluster3 = np.random.randn(50, 2) + np.array([0, 5])

    X = np.vstack([cluster1, cluster2, cluster3])

    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]

    # Fit K-means
    print("Training K-Means Clustering (K=3)...")
    kmeans = KMeans(n_clusters=3, max_iters=100, random_state=42)
    kmeans.fit(X)

    print(f"\nFinal Centroids:")
    for i, centroid in enumerate(kmeans.centroids):
        print(f"Cluster {i}: {centroid}")

    print(f"\nCluster sizes:")
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"Cluster {cluster_id}: {count} points")

    # Test prediction on new points
    X_test = np.array([[0, 0], [5, 5], [0, 5], [2.5, 2.5]])
    predictions = kmeans.predict(X_test)

    print(f"\nTest Predictions:")
    for i, point in enumerate(X_test):
        print(f"Point {point}: Assigned to Cluster {predictions[i]}")
