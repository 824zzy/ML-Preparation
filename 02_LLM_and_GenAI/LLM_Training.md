# LLM Training

## Pre-training Objectives

### Causal Language Modeling (CLM)
- Predict next token given previous tokens
- Used by GPT models
- Loss: cross-entropy on next token prediction
- Naturally suited for text generation

### Masked Language Modeling (MLM)
- Randomly mask tokens, predict them from context
- Used by BERT
- Bidirectional context (can see tokens before and after)
- Better for understanding tasks, but doesn't naturally generate

### Denoising
- Corrupt input (mask/delete/shuffle), reconstruct original
- Used by T5, BART
- More general than MLM (can corrupt in multiple ways)
- Good for seq2seq tasks

**Modern trend:** CLM on decoder-only models (GPT-style) dominates because it's simpler and scales better.

## Supervised Fine-Tuning (SFT)

**What:** Fine-tune pre-trained model on instruction-following examples.

**Process:**
1. Start with pre-trained base model
2. Create dataset of (instruction, response) pairs
3. Fine-tune with standard language modeling loss
4. Model learns to follow instructions

**Dataset examples:**
- FLAN collection (chain-of-thought, dialogue, Q&A)
- ShareGPT (human conversations)
- Self-instruct (generated from existing LLM)

**Key insight:** Pre-training gives knowledge, SFT teaches the model how to use it conversationally.

**Typical scale:** 50k-1M examples, 1-3 epochs.

## RLHF (Reinforcement Learning from Human Feedback)

**Three-stage pipeline:**

### Stage 1: Train Reward Model
1. Sample prompts, generate multiple completions from SFT model
2. Humans rank completions (best to worst)
3. Train reward model to predict human preferences
4. Loss: pairwise ranking loss

**Reward model:** Usually initialized from SFT model, trains a scalar output head.

### Stage 2: RL Fine-tuning
1. Use reward model as reward function
2. Optimize policy (LLM) with PPO (Proximal Policy Optimization)
3. Add KL penalty to prevent drift from SFT model

**Objective:**
```
maximize E[r(x, y)] - β * KL(π_θ || π_ref)
```
- r(x, y): reward model score
- π_θ: policy being optimized
- π_ref: reference policy (SFT model)
- β: KL penalty coefficient

### Stage 3: Iterate
- Deploy model, collect more human feedback
- Retrain reward model and policy

**Why KL penalty?**
- Prevents mode collapse (model gaming the reward)
- Preserves general capabilities from pre-training
- Keeps outputs reasonable

**Challenges:**
- Expensive: need human raters
- Reward hacking: model exploits reward model
- Complex: RL is unstable, requires careful tuning

## DPO (Direct Preference Optimization)

**Key insight:** Skip the reward model and RL. Optimize preferences directly.

**Formula:**
```
L_DPO = -E[log σ(β log π_θ(y_w|x) / π_ref(y_w|x) - β log π_θ(y_l|x) / π_ref(y_l|x))]
```

Where:
- y_w: preferred (winning) response
- y_l: dispreferred (losing) response
- π_θ: policy being trained
- π_ref: reference policy (SFT model)
- β: temperature parameter
- σ: sigmoid function

**Intuition:** Increase probability of preferred responses relative to dispreferred ones, while staying close to reference policy.

**Advantages over RLHF:**
- Simpler: no reward model, no RL
- More stable: standard supervised learning
- Faster: one training stage instead of three

**Disadvantages:**
- Less flexible: can't easily incorporate non-pairwise feedback
- Relies on high-quality preference data

**When to use:** DPO is becoming the default for alignment. Use RLHF when you need complex reward functions or online learning.

## Constitutional AI (Anthropic)

**Goal:** Make models helpful, harmless, and honest without extensive human feedback.

**Two stages:**

### Stage 1: Supervised Learning (SL-CAI)
1. Model generates responses to harmful prompts
2. Model critiques and revises its own responses using constitutional principles
3. Train on (prompt, revised response) pairs

**Constitutional principles:** Rules like "responses should not be racist" or "choose the response that is most helpful and harmless."

### Stage 2: RL from AI Feedback (RLAIF)
1. Model generates pairs of responses
2. Model evaluates which is better according to constitution
3. Train reward model on AI preferences
4. Use reward model for RL (like RLHF)

**Key difference from RLHF:** AI provides feedback instead of humans, using explicit principles.

**Advantages:**
- Scales better (less human labor)
- More transparent (principles are explicit)
- Reduces human bias in feedback

**Used by:** Claude models.

## Scaling Laws

**Q: How should we allocate compute between model size and training data?**

### Chinchilla Scaling Laws (DeepMind, 2022)

**Key finding:** For compute-optimal training, model size and training tokens should scale equally.

**Chinchilla formula:**
- For a compute budget C, optimal model has N parameters and should be trained on D ≈ 20N tokens
- Previous models (GPT-3) were over-parameterized and under-trained

**Implications:**
- GPT-3 (175B params, 300B tokens) was not compute-optimal
- Chinchilla (70B params, 1.4T tokens) outperformed GPT-3 with less compute
- LLaMA models follow this: smaller but trained on more data

**Practical takeaway:** Don't just make models bigger. Train them longer on more data.

### Other Scaling Laws

**Kaplan et al. (OpenAI, 2020):**
- Earlier scaling laws (pre-Chinchilla)
- Suggested training on ~200B tokens regardless of model size

**Loss scaling:**
- Test loss scales as power law with compute: L(C) ∝ C^(-α)
- Predictable performance from smaller models

**Emergent abilities:**
- Some capabilities appear suddenly at scale (few-shot learning, chain-of-thought)
- Not smoothly predictable from smaller models

## Data Curation and Quality

**Q: How important is training data quality?**

Extremely. "Garbage in, garbage out" applies at massive scale.

### Pre-training Data

**Sources:**
- Web crawl (CommonCrawl)
- Books (Books3, Project Gutenberg)
- Code (GitHub, StackOverflow)
- Wikipedia, academic papers

**Curation strategies:**
1. **Deduplication:** Remove near-duplicates (improves efficiency and reduces memorization)
2. **Filtering:** Remove low-quality pages (classifier-based, heuristics)
3. **Toxic content removal:** Filter harmful content
4. **PII removal:** Scrub personal information
5. **Language filtering:** Keep target languages
6. **Domain balancing:** Mix domains (code, math, web) in desired proportions

**Quality proxy:** Train a classifier to distinguish high-quality sources (Wikipedia, books) from low-quality (random web pages), then filter by score.

### Fine-tuning Data

**Instruction data:**
- Diversity matters more than size
- Cover many task types and styles
- Human-written > model-generated (but expensive)

**RLHF data:**
- Preference pairs must be clear (not too similar)
- Raters need clear guidelines
- Diversity in prompts critical

**Contamination:** Test sets leaking into training data is a major issue. Hard to detect, can inflate benchmark scores artificially.

## Training Infrastructure

**Typical large model training:**
- Thousands of GPUs/TPUs
- Distributed training (data parallel + model parallel + pipeline parallel)
- Mixed precision (FP16/BF16 with FP32 master weights)
- Gradient checkpointing (trade compute for memory)
- ZeRO optimizer (shards optimizer states across devices)

**Stability tricks:**
- Learning rate warmup
- Gradient clipping
- Pre-norm architecture
- Monitoring loss spikes and restarting from checkpoints

**Training time:** GPT-3 took ~3-4 months on ~10k V100 GPUs. Modern models similar or longer.
