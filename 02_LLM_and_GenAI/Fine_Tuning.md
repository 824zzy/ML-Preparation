# Fine-Tuning

## Full Fine-Tuning vs PEFT

### Full Fine-Tuning
- Update all model parameters
- Requires storing gradients and optimizer states for all parameters
- Memory: ~4x model size (model + gradients + optimizer states)
- For 7B model: ~28GB just for parameters during training
- Best performance, but expensive

### Parameter-Efficient Fine-Tuning (PEFT)
- Update only a small subset of parameters
- Much lower memory and compute
- Slight performance drop vs full fine-tuning
- Can fine-tune LLMs on consumer GPUs

**When to use full vs PEFT:**
- Full: You have compute budget and need absolute best performance
- PEFT: Limited compute, good enough performance, or need many task-specific adapters

## LoRA (Low-Rank Adaptation)

**Key idea:** Model updates live in a low-rank subspace. Instead of updating weight matrix W, add a low-rank decomposition.

**Formula:**
```
W' = W + BA

Where:
- W: original weight (frozen)
- B: d × r matrix
- A: r × k matrix
- r: rank (typically 8, 16, or 32)
```

**Parameters:** Instead of d × k, train only (d + k) × r parameters. If r << min(d, k), huge savings.

**Example:** For a 4096 × 4096 weight matrix:
- Full fine-tuning: 16M parameters
- LoRA with r=16: (4096 + 4096) × 16 = 131k parameters (~100x reduction)

**Which layers to apply LoRA?**
- Original paper: just Q and V projection in attention
- Common practice: Q, K, V, and output projection
- Sometimes: all linear layers including FFN

**Advantages:**
- Much lower memory (no optimizer states for frozen weights)
- Can train multiple task-specific adapters, swap at inference
- Faster training

**Disadvantages:**
- Slightly worse than full fine-tuning
- Need to tune rank r (higher r = better but more expensive)

**Merging:** After training, can merge W + BA into single matrix for inference (no overhead).

## QLoRA (Quantized LoRA)

**Key idea:** Combine LoRA with quantization. Store base model in 4-bit, train LoRA adapters in higher precision.

**Process:**
1. Load pre-trained model in 4-bit (NormalFloat4)
2. Freeze quantized weights
3. Add LoRA adapters (trained in BF16)
4. Backprop through quantized weights (forward only), update adapters

**Memory savings:**
- 65B model fits in 48GB GPU memory
- Can fine-tune 33B model on single A100 (40GB)

**NormalFloat4 (NF4):**
- Quantization scheme optimized for normally distributed weights
- Better than standard INT4

**Double quantization:** Also quantize the quantization constants (saves ~0.5 bit per parameter).

**Performance:** Near-parity with full 16-bit fine-tuning on many tasks.

**When to use:** You want to fine-tune large models but don't have multiple GPUs.

## Other PEFT Methods

### Prefix Tuning
- Prepend learnable "prefix" tokens to input
- Only train prefix embeddings, freeze model
- Fewer parameters than LoRA but often worse performance

### Adapters
- Insert small bottleneck layers between transformer blocks
- Train adapter layers, freeze original model
- More parameters than LoRA (need to store extra layers)

### (IA)³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
- Learn scalar multipliers for activations
- Even fewer parameters than LoRA
- Competitive performance on some tasks

**Ranking by memory efficiency:**
1. (IA)³ (most efficient)
2. LoRA
3. Prefix tuning
4. Adapters
5. Full fine-tuning (least efficient)

**Ranking by performance:**
1. Full fine-tuning (best)
2. LoRA / QLoRA
3. Adapters
4. Prefix tuning
5. (IA)³ (varies by task)

**Industry standard:** LoRA/QLoRA for most use cases.

## Quantization

**Goal:** Reduce model size for faster inference and lower memory.

### Post-Training Quantization (PTQ)

**INT8:**
- 8-bit integers instead of 16/32-bit floats
- 2-4x speedup on appropriate hardware
- Minimal accuracy loss for most models
- Widely supported (TensorRT, ONNX, PyTorch)

**INT4:**
- 4-bit integers
- 4x memory reduction vs FP16
- Slightly more accuracy loss than INT8
- Used in QLoRA, GPTQ, AWQ

**Calibration:** Use small dataset to find optimal quantization ranges.

### GPTQ (GPT Quantization)
- Layer-wise quantization algorithm
- Minimizes reconstruction error per layer
- Slower quantization (takes hours for large models)
- Good quality 4-bit models

### AWQ (Activation-aware Weight Quantization)
- Observes which weights are important based on activations
- Protects important weights from quantization error
- Faster quantization than GPTQ
- Often better quality at same bit-width

### GGUF/GGML (LLaMA.cpp format)
- CPU-friendly quantization (2-8 bits)
- Designed for running LLMs on CPUs and consumer GPUs
- Variable bit-width (mix of 4-bit, 5-bit, 6-bit)
- Popular for local LLM inference

**Quantization-Aware Training (QAT):**
- Simulate quantization during training
- Better accuracy than PTQ
- But more expensive (need to retrain)

**Practical advice:**
- INT8: near-zero quality loss, do it by default
- INT4 with GPTQ/AWQ: good for 7B+ models, test on your task
- Below 4-bit: usually too much quality degradation

## When to Fine-Tune vs Prompt Engineer vs RAG

### Prompt Engineering
**Use when:**
- Task is within model's existing capabilities
- You need fast iteration
- You have few examples (<100)
- Task will change frequently

**Examples:** Classification, simple extraction, reformatting.

### RAG (Retrieval-Augmented Generation)
**Use when:**
- Model needs access to external knowledge
- Knowledge changes frequently (can't retrain)
- Task is question-answering over documents
- You want citations/sources

**Examples:** Q&A over company docs, customer support with knowledge base.

### Fine-Tuning
**Use when:**
- Need specific style, format, or domain knowledge
- Have enough data (1k+ examples)
- Task doesn't change frequently
- Prompt engineering isn't good enough
- Want lower latency (shorter prompts)

**Examples:** Domain-specific generation (legal, medical), brand voice, complex reasoning patterns.

### Decision flowchart:
1. **Can the base model do it with good prompts?** → Prompt engineering
2. **Does it need external knowledge that changes?** → RAG
3. **Does it need specific behavior or domain expertise?** → Fine-tuning
4. **Need external knowledge AND specific behavior?** → RAG + fine-tuning

**Cost considerations:**
- Prompt engineering: free (just inference cost)
- RAG: inference cost + embedding/indexing cost
- Fine-tuning: training cost (one-time) + inference cost

**Latency:**
- Prompt engineering: moderate (can have long prompts)
- RAG: high (retrieval + long context)
- Fine-tuning: low (short prompts)

**Common mistake:** Fine-tuning when prompt engineering would work. Try prompting first.

## Fine-Tuning Best Practices

**Data quality over quantity:**
- 1k high-quality examples > 10k mediocre ones
- Diverse examples covering edge cases
- Clean formatting (consistent instruction format)

**Hyperparameters:**
- Learning rate: 1e-5 to 5e-5 (much lower than pre-training)
- Epochs: 3-5 (too many causes overfitting)
- Batch size: as large as memory allows (with gradient accumulation)
- Warmup: 5-10% of steps

**Monitoring:**
- Track loss on validation set
- Generate samples every N steps (check quality)
- Watch for catastrophic forgetting (test on general benchmarks)

**Avoiding catastrophic forgetting:**
- Don't train too long
- Use lower learning rates
- Mix in general instruction data
- Use LoRA (limits how much model can change)

**Instruction format consistency:**
```
# Example format
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

Keep format identical between training and inference.

## Hands-On: NanoChat

To see supervised fine-tuning in practice:

**SFT pipeline**: Check `scripts/chat_sft.py` for the full supervised fine-tuning implementation. Shows how to take a pretrained base model and fine-tune it on instruction-following data. Notice how simple the process is compared to RLHF (just standard language modeling loss on instruction-response pairs).
