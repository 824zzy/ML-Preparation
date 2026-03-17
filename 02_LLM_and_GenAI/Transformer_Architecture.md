# Transformer Architecture

## Self-Attention Mechanism

**Q: What is self-attention?**

A mechanism that lets each token in a sequence attend to all other tokens to compute contextual representations.

**Formula:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q (query), K (key), V (value) are linear projections of input embeddings
- d_k is the dimension of the key vectors
- softmax is applied row-wise, producing attention weights

**Process:**
1. Compute similarity scores between query and all keys: QK^T
2. Scale by √d_k
3. Apply softmax to get attention weights (sum to 1)
4. Weighted sum of values

**Q: Why is self-attention better than RNNs?**

- Parallelizable: all positions computed simultaneously (RNNs are sequential)
- Direct connections: O(1) path length between any two positions (RNNs are O(n))
- Better long-range dependencies: no vanishing gradient through time
- More interpretable: can visualize attention weights

**Downside:** O(n²) complexity in sequence length (vs O(n) for RNNs).

## Scaling by √d_k

**Q: Why scale by √d_k?**

Without scaling, dot products grow large when d_k is large, pushing softmax into regions with tiny gradients.

**Intuition:** If Q and K have elements drawn from N(0,1), their dot product has variance d_k. Dividing by √d_k normalizes variance to 1, keeping softmax gradients healthy.

**What happens without scaling?** Softmax saturates (all weight on one token), making gradients vanish and training unstable.

## Multi-Head Attention

**Q: What is multi-head attention?**

Run multiple attention operations in parallel with different learned projections, then concatenate results.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Q: Why use multiple heads?**

- Different heads learn different patterns (syntax, semantics, positional relationships)
- Analogous to multiple CNN filters learning different features
- Increases model capacity without increasing per-head dimension

**Typical setup:** 8-16 heads, d_model = 512, d_k = d_v = 64 per head.

## Positional Encoding

**Q: Why do we need positional encoding?**

Self-attention is permutation-invariant. Without position info, "dog bites man" and "man bites dog" look identical.

**Three approaches:**

### 1. Sinusoidal (Original Transformer)

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- Deterministic, no parameters
- Can extrapolate to longer sequences than seen during training
- Used in original Transformer

### 2. Learned Positional Embeddings

- Each position gets a learned embedding vector
- Used in BERT, GPT-2
- Better performance on fixed-length sequences
- Cannot extrapolate beyond training length

### 3. Rotary Position Embedding (RoPE)

- Applies rotation matrix to Q and K based on position
- Relative position info encoded in dot product
- Better extrapolation to longer sequences
- Used in modern LLMs (GPT-NeoX, PaLM, LLaMA)

**RoPE wins** for models that need to handle variable or long contexts.

## Encoder-Decoder vs Decoder-Only

### Encoder-Decoder (e.g., Original Transformer, T5)

**Structure:**
- Encoder: bidirectional attention on input
- Decoder: causal attention on output, cross-attention to encoder

**Use case:** Seq2seq tasks (translation, summarization where input and output are different)

**Example: BERT**
- Encoder-only (no decoder)
- Bidirectional attention
- Pre-trained with masked language modeling (MLM)
- Good for understanding tasks (classification, NER, Q&A)

### Decoder-Only (e.g., GPT, LLaMA, Claude)

**Structure:**
- Single stack with causal (autoregressive) attention
- Each token can only attend to previous tokens

**Use case:** Text generation, few-shot learning

**Why decoder-only won?**
- Simpler architecture
- Scales better
- Can do both generation and understanding (with prompting)
- Most modern LLMs (GPT-3/4, PaLM, LLaMA, Claude) are decoder-only

## Layer Normalization

**Q: Pre-norm vs post-norm?**

### Post-Norm (Original Transformer)
```
x = x + Sublayer(LayerNorm(x))
```

### Pre-Norm
```
x = x + LayerNorm(Sublayer(x))
```

**Pre-norm advantages:**
- More stable training (especially for deep models)
- Can train without warmup
- Used in most modern LLMs (GPT-3, LLaMA)

**Post-norm advantages:**
- Slightly better performance when training is stable
- Original Transformer used this

**In practice:** Pre-norm is now standard for LLMs.

## FlashAttention

**Q: What is FlashAttention?**

An algorithm that computes exact attention with fewer memory reads/writes by fusing operations and using tiling.

**Standard attention problem:**
- Materializes full n × n attention matrix
- For long sequences, doesn't fit in fast GPU memory (SRAM)
- Requires many slow HBM (GPU DRAM) reads/writes

**FlashAttention solution:**
- Tiles the computation into blocks that fit in SRAM
- Fuses softmax, dropout, masking into one kernel
- Never materializes full attention matrix
- 2-4x faster, uses less memory

**Why it matters:**
- Enables training on longer sequences
- Faster inference
- Standard in modern LLM implementations (PyTorch 2.0+, Triton)

## KV Cache

**Q: What is KV cache?**

During autoregressive generation, cache the key and value projections of previous tokens to avoid recomputing them.

**Without KV cache:**
- To generate token t, recompute K and V for all tokens 1..t-1
- O(n²) computation for sequence length n

**With KV cache:**
- Store K and V from previous steps
- Only compute K and V for new token
- O(n) computation

**Memory tradeoff:**
- Saves computation
- Uses O(n × d_model × num_layers) memory
- For long contexts, KV cache can be the memory bottleneck

**Optimizations:**
- Multi-query attention (MQA): share K/V across heads
- Grouped-query attention (GQA): share K/V across head groups
- Both reduce KV cache size

## Context Window Scaling

**Q: How to extend context windows beyond training length?**

### 1. Position Interpolation (PI)
- Scale position indices to fit within training range
- Example: 4k training → 8k inference, scale positions by 0.5
- Simple, works surprisingly well

### 2. YaRN (Yet another RoPE extensioN)
- Improved position interpolation with temperature scaling
- Better preserves high-frequency information

### 3. ALiBi (Attention with Linear Biases)
- Add learned linear bias to attention scores based on distance
- No positional encodings needed
- Extrapolates well to longer sequences

### 4. Sliding Window Attention
- Only attend to fixed window of recent tokens
- Reduces complexity from O(n²) to O(n × w)
- Used in Mistral models

### 5. Sparse Attention Patterns
- Longformer: local + global attention
- BigBird: local + global + random attention
- Reduces complexity while maintaining long-range connections

**Most practical for extending existing models:** Position interpolation or fine-tuning with longer sequences.
