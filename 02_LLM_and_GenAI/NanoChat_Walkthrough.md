# NanoChat Walkthrough

A learning guide for [NanoChat](https://github.com/karpathy/nanochat) as a practical companion to the theoretical LLM content in this repo.

## What is NanoChat?

Karpathy's minimal end-to-end LLM training framework. You can train a GPT-2-capable model for around $48. It covers the full pipeline that production LLMs go through:

- Tokenization (BPE)
- Pretraining (causal language modeling)
- Supervised fine-tuning (SFT)
- Reinforcement learning from human feedback (RLHF)
- Evaluation (MMLU, GSM8K, ARC, DCLM CORE)
- Efficient inference (KV cache)

Unlike nanoGPT (which focuses on pretraining), NanoChat shows you the entire pipeline from base model to chat model.

## Repository Map

Each file teaches you something specific. Here's what to focus on:

| File | What It Teaches | Why It Matters |
|------|-----------------|----------------|
| `nanochat/gpt.py` | Transformer implementation (attention, layer norm, MLP, residual) | See how attention actually works in code. Good reference for "implement attention" questions. |
| `nanochat/tokenizer.py` | BPE tokenization | Understand how text becomes tokens. Important for debugging tokenization issues. |
| `nanochat/dataloader.py` | Distributed data loading | How to efficiently load training data at scale. Handles sharding across GPUs. |
| `nanochat/engine.py` | Inference with KV cache | See how autoregressive generation is optimized. Crucial for understanding inference speed. |
| `nanochat/optim.py` | AdamW and Muon optimizers | Two modern optimizers. AdamW is standard, Muon is a newer alternative. |
| `scripts/base_train.py` | Pretraining pipeline | End-to-end training loop: data loading, forward/backward pass, optimization, checkpointing. |
| `scripts/chat_sft.py` | Supervised fine-tuning | How to take a base model and teach it to follow instructions. Same pipeline as ChatGPT's first stage. |
| `scripts/chat_rl.py` | RLHF training | Full RLHF implementation with reward model and PPO. Second stage of ChatGPT training. |
| `tasks/mmlu.py` | MMLU benchmark | How benchmarks work: prompt formatting, parsing, scoring. |
| `tasks/gsm8k.py` | GSM8K (math reasoning) | Chain-of-thought evaluation. See how to extract numeric answers. |
| `tasks/arc.py` | ARC (science reasoning) | Multiple choice evaluation. Similar to MMLU but different domain. |
| `nanochat/core_eval.py` | DCLM CORE evaluation | Pretraining quality metric. Measures model capability during base training. |
| `runs/scaling_laws.sh` | Scaling law experiments | Scripts for training models of different sizes. Makes scaling laws concrete. |

## Key Things to Study

When reading the code, focus on these concepts:

### 1. Attention Implementation (`nanochat/gpt.py`)

Look for:
- How Q, K, V are computed (linear projections)
- The scaled dot-product attention formula: `softmax(QK^T / √d_k) V`
- Multi-head attention (splitting and concatenating heads)
- Causal masking (upper triangular mask for autoregressive generation)
- RoPE (Rotary Position Embedding) for position encoding

This answers the interview question: "Implement attention from scratch."

### 2. KV Cache (`nanochat/engine.py`)

Look for:
- How keys and values are cached across generation steps
- Why this turns O(n²) generation into O(n)
- Memory vs compute tradeoff

This answers: "How does KV cache speed up inference?"

### 3. The `--depth` Parameter

NanoChat uses a single complexity dial:
- `--depth`: Controls model size (number of layers, hidden dim, etc.)
- Higher depth = bigger model, more capable, more expensive
- Makes it easy to experiment with different model sizes

This is relevant for understanding scaling laws.

### 4. Mixed Precision Training

Look for:
- Use of `bfloat16` for forward/backward pass
- Storing optimizer state in higher precision
- Why this speeds up training and reduces memory

Standard in modern LLM training.

### 5. The SFT → RL Pipeline (`scripts/chat_sft.py`, `scripts/chat_rl.py`)

Study the two-stage process:
1. **SFT**: Fine-tune base model on (instruction, response) pairs. Standard supervised learning.
2. **RL**: Optimize for human preferences using reward model and PPO.

This is exactly how ChatGPT and Claude are trained.

### 6. How Evaluation Benchmarks Work (`tasks/`)

Look at how benchmarks are structured:
- Prompt formatting (how the question is presented)
- Answer extraction (parsing model output)
- Scoring (exact match, multiple choice, numeric comparison)

Helps you understand what benchmarks actually measure.

## Interview Relevance

NanoChat helps you answer these common interview questions:

| Interview Question | Where to Look |
|--------------------|---------------|
| "Implement attention from scratch" | `nanochat/gpt.py` (CausalSelfAttention class) |
| "Walk me through a training loop" | `scripts/base_train.py` (main training loop) |
| "How does KV cache speed up inference?" | `nanochat/engine.py` (generate function) |
| "Explain the RLHF pipeline" | `scripts/chat_sft.py` + `scripts/chat_rl.py` |
| "What are scaling laws?" | `runs/scaling_laws.sh` (experiments with different model sizes) |
| "How do you implement MMLU?" | `tasks/mmlu.py` (prompt formatting and scoring) |
| "What optimizations make LLM training faster?" | Mixed precision, gradient accumulation, distributed training (all in `scripts/base_train.py`) |
| "How does BPE tokenization work?" | `nanochat/tokenizer.py` |
| "What's the difference between SFT and RLHF?" | Compare `scripts/chat_sft.py` (standard LM loss) vs `scripts/chat_rl.py` (reward + PPO) |

## How to Use This for Interview Prep

1. **Read the code alongside the theory**: As you study Transformer_Architecture.md, open `nanochat/gpt.py` to see it implemented.

2. **Trace through the training pipeline**: Follow a training run from data loading → forward pass → loss → backward pass → optimizer step. This is in `scripts/base_train.py`.

3. **Understand the full ChatGPT pipeline**: Go through `scripts/base_train.py` (pretraining) → `scripts/chat_sft.py` (instruction tuning) → `scripts/chat_rl.py` (RLHF). You'll understand how production LLMs are built.

4. **Study one benchmark deeply**: Pick `tasks/mmlu.py` or `tasks/gsm8k.py`. Understand prompt formatting, answer extraction, scoring. This helps you design evaluation for system design rounds.

5. **Implement attention yourself**: After reading `nanochat/gpt.py`, implement it from scratch without looking. This preps you for ML coding rounds.

## What Makes NanoChat Good for Learning

**Minimal but complete**: Unlike production codebases (HuggingFace Transformers, Megatron), NanoChat is short enough to read in an afternoon but complete enough to train real models.

**No abstraction hell**: Production frameworks hide details behind layers of abstraction. NanoChat shows you exactly what's happening.

**Modern techniques**: Uses current best practices (mixed precision, distributed training, KV cache, RoPE). Not outdated tutorial code.

**End-to-end**: Most tutorials stop at pretraining. NanoChat shows you SFT and RLHF too, which is what you need for production chat models.

## Suggested Study Order

1. Start with `nanochat/gpt.py` (transformer architecture)
2. Read `nanochat/engine.py` (inference with KV cache)
3. Skim `scripts/base_train.py` (training loop structure)
4. Read `scripts/chat_sft.py` (supervised fine-tuning)
5. Read `scripts/chat_rl.py` (RLHF)
6. Pick one benchmark from `tasks/` to study in detail
7. (Optional) Read `nanochat/tokenizer.py` if tokenization comes up

Total reading time: 3-4 hours. Then refer back as needed during interview prep.

## Connection to Other Files

- **Transformer_Architecture.md**: Theory for what's implemented in `nanochat/gpt.py`
- **LLM_Training.md**: Describes the pipelines implemented in `scripts/`
- **Fine_Tuning.md**: Theory for `scripts/chat_sft.py`
- **Evaluation.md**: Context for `tasks/` benchmarks
- **Safety_and_Alignment.md**: Background for why RLHF (`scripts/chat_rl.py`) is needed
