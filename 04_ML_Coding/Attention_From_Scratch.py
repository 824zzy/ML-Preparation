"""
Attention Mechanism from Scratch using NumPy

Implements scaled dot-product attention and multi-head attention.

Key formulas:
- Scaled Dot-Product Attention:
  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V

  where:
  - Q: Query matrix (n_queries, d_k)
  - K: Key matrix (n_keys, d_k)
  - V: Value matrix (n_keys, d_v)
  - d_k: dimension of keys/queries

- Multi-Head Attention:
  1. Project Q, K, V into h different subspaces
  2. Apply attention in each subspace in parallel
  3. Concatenate results and project back
"""

import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class ScaledDotProductAttention:
    """Scaled dot-product attention mechanism."""

    def __call__(self, Q, K, V, mask=None):
        """
        Apply scaled dot-product attention.

        Args:
            Q: Query matrix (batch_size, n_queries, d_k)
            K: Key matrix (batch_size, n_keys, d_k)
            V: Value matrix (batch_size, n_keys, d_v)
            mask: Optional mask (batch_size, n_queries, n_keys)

        Returns:
            output: Attention output (batch_size, n_queries, d_v)
            attention_weights: Attention weights (batch_size, n_queries, n_keys)
        """
        d_k = K.shape[-1]

        # Calculate attention scores: QK^T / sqrt(d_k)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

        # Apply mask if provided (set masked positions to -inf)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Apply softmax to get attention weights
        attention_weights = softmax(scores, axis=-1)

        # Apply attention to values
        output = np.matmul(attention_weights, V)

        return output, attention_weights


class MultiHeadAttention:
    """Multi-head attention mechanism."""

    def __init__(self, d_model, n_heads):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Initialize projection matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x):
        """
        Split the last dimension into (n_heads, d_k).

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Reshaped tensor (batch_size, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        """
        Combine heads back to d_model.

        Args:
            x: Input tensor (batch_size, n_heads, seq_len, d_k)

        Returns:
            Combined tensor (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)

    def __call__(self, Q, K, V, mask=None):
        """
        Apply multi-head attention.

        Args:
            Q: Query matrix (batch_size, seq_len_q, d_model)
            K: Key matrix (batch_size, seq_len_k, d_model)
            V: Value matrix (batch_size, seq_len_v, d_model)
            mask: Optional mask

        Returns:
            output: Attention output (batch_size, seq_len_q, d_model)
            attention_weights: Average attention weights across heads
        """
        batch_size = Q.shape[0]

        # Linear projections
        Q = np.matmul(Q, self.W_q)
        K = np.matmul(K, self.W_k)
        V = np.matmul(V, self.W_v)

        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch_size, n_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (batch_size, n_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (batch_size, n_heads, seq_len_v, d_k)

        # Apply attention for each head
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # Combine heads
        output = self.combine_heads(attn_output)

        # Final linear projection
        output = np.matmul(output, self.W_o)

        # Average attention weights across heads for visualization
        avg_attn_weights = np.mean(attn_weights, axis=1)

        return output, avg_attn_weights


if __name__ == "__main__":
    np.random.seed(42)

    # Example: attention for a simple sequence
    batch_size = 2
    seq_len = 4
    d_model = 8
    n_heads = 2

    # Create sample input (batch_size, seq_len, d_model)
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)

    print("=" * 60)
    print("Scaled Dot-Product Attention")
    print("=" * 60)

    attention = ScaledDotProductAttention()
    output, weights = attention(Q, K, V)

    print(f"Input shape: {Q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"\nAttention weights (first sample):")
    print(weights[0])

    print("\n" + "=" * 60)
    print("Multi-Head Attention")
    print("=" * 60)

    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    output_mha, avg_weights = mha(Q, K, V)

    print(f"Input shape: {Q.shape}")
    print(f"Output shape: {output_mha.shape}")
    print(f"Average attention weights shape: {avg_weights.shape}")
    print(f"\nAverage attention weights across heads (first sample):")
    print(avg_weights[0])

    # Demonstrate masking (e.g., for causal/autoregressive attention)
    print("\n" + "=" * 60)
    print("Causal Attention (with mask)")
    print("=" * 60)

    # Create causal mask (lower triangular matrix)
    causal_mask = np.tril(np.ones((batch_size, seq_len, seq_len)))
    output_masked, weights_masked = attention(Q, K, V, mask=causal_mask)

    print(f"Causal attention weights (first sample):")
    print(weights_masked[0])
    print("\nNote: Upper triangle is zero due to causal masking")
