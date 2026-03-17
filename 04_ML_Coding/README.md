# ML Coding

## What This Tests

ML coding interviews test your ability to implement ML algorithms from scratch, without using high-level libraries like scikit-learn or PyTorch. You'll be given a coding environment with only NumPy and basic Python.

**Key skills tested:**
- Understanding of algorithm internals (not just API usage)
- NumPy proficiency (vectorization, broadcasting)
- Numerical stability (avoid overflow, underflow, division by zero)
- Code clarity and correctness
- Ability to explain your implementation

## Format

Typically 45-60 minutes. Common patterns:
1. Implement a classic ML algorithm (linear regression, k-means, decision tree)
2. Implement a loss function or evaluation metric
3. Implement a data preprocessing step (normalization, one-hot encoding)
4. Debug or optimize existing ML code

## Implementations in This Directory

### Supervised Learning
- `linear_regression.py` - Closed-form solution and gradient descent (Easy)
- `logistic_regression.py` - Binary classification with gradient descent (Medium)
- `naive_bayes.py` - Gaussian and Multinomial Naive Bayes (Easy)
- `k_nearest_neighbors.py` - Classification and regression (Easy)
- `decision_tree.py` - CART algorithm for classification (Hard)
- `svm.py` - Linear SVM with hinge loss (Hard)

### Unsupervised Learning
- `k_means.py` - Standard k-means clustering (Medium)
- `pca.py` - Principal Component Analysis via SVD (Medium)
- `gmm.py` - Gaussian Mixture Model with EM algorithm (Hard)

### Neural Networks
- `neural_network.py` - Simple feedforward network with backprop (Hard)
- `activation_functions.py` - Sigmoid, ReLU, softmax, tanh (Easy)
- `loss_functions.py` - MSE, cross-entropy, hinge (Easy)

### Metrics & Utilities
- `metrics.py` - Accuracy, precision, recall, F1, AUC (Easy)
- `cross_validation.py` - K-fold cross-validation (Medium)
- `regularization.py` - L1, L2 regularization (Easy)
- `normalization.py` - Min-max, z-score, batch norm (Easy)

### Optimization
- `gradient_descent.py` - Vanilla, SGD, mini-batch (Medium)
- `adam.py` - Adam optimizer (Medium)

## Key Implementation Tips

### Use NumPy Only
```python
# ✅ Good
import numpy as np
predictions = 1 / (1 + np.exp(-z))

# ❌ Bad (don't use these in interviews)
from sklearn.linear_model import LogisticRegression
import torch
```

### Vectorize Everything
```python
# ✅ Good (vectorized)
distances = np.sqrt(((X - point) ** 2).sum(axis=1))

# ❌ Bad (slow loop)
distances = []
for i in range(len(X)):
    distances.append(np.sqrt(np.sum((X[i] - point) ** 2)))
```

### Handle Numerical Stability
```python
# ✅ Good (log-sum-exp trick)
def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ❌ Bad (overflow risk)
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))
```

### Check Shapes Often
```python
# Add assertions to catch bugs early
assert X.shape == (n_samples, n_features)
assert y.shape == (n_samples,)
assert weights.shape == (n_features,)
```

### Add Docstrings
```python
def fit(self, X, y):
    """
    Fit the model to training data.

    Args:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)

    Returns:
        self (for method chaining)
    """
    # implementation
```

## Common Pitfalls

**Broadcasting mistakes:**
```python
# Be explicit about dimensions
# (n, 1) vs (n,) behave differently in broadcasting
```

**Not initializing properly:**
```python
# ✅ Random initialization for neural networks
weights = np.random.randn(n_features, n_classes) * 0.01

# ❌ Zero initialization (bad for neural networks)
weights = np.zeros((n_features, n_classes))
```

**Forgetting bias term:**
```python
# ✅ Remember to add bias
prediction = X @ weights + bias

# ❌ Missing bias
prediction = X @ weights
```

**Off-by-one errors in loops:**
```python
# When implementing gradient descent, make sure you're updating
# the right number of times (iterations vs epochs)
```

**Not handling edge cases:**
```python
# What if k > n_samples in k-NN?
# What if variance is zero in normalization?
# What if log(0) in cross-entropy?
```

## Interview Strategy

1. **Clarify requirements:**
   - Input/output shapes?
   - Should I handle edge cases?
   - Do you want closed-form or iterative solution?

2. **Start with simple version:**
   - Get basic working code first
   - Optimize later if time permits

3. **Test as you go:**
   - Use small toy examples
   - Check intermediate shapes

4. **Explain your approach:**
   - Walk through the math
   - Explain vectorization choices
   - Discuss trade-offs (speed vs memory)

5. **Common optimizations to mention:**
   - Vectorization (no for loops)
   - Caching (store computed values)
   - Early stopping (convergence check)
   - Batch processing (for large datasets)

## Time Management

- 5 min: Understand problem, clarify requirements
- 10 min: Write skeleton code, define function signatures
- 20 min: Implement core algorithm
- 10 min: Test with examples, fix bugs
- 5 min: Discuss optimizations, complexity

## Resources

- NumPy documentation (know broadcasting rules, axis parameter, keepdims)
- Practice on paper first (derive gradients, understand math)
- Implement from scratch at least once (even if you know sklearn)
