# LLM and GenAI

This section covers LLM/GenAI knowledge that now shows up in 60%+ of ML interviews. Focus on understanding the trade-offs and when to use each approach.

## Hands-On Companion: NanoChat

For practical understanding, see [NanoChat](https://github.com/karpathy/nanochat), Karpathy's minimal LLM training framework. It covers the full pipeline: tokenization, pretraining, SFT, RLHF, evaluation, and inference with KV cache. Use it as a hands-on reference while studying this section. See **NanoChat_Walkthrough.md** for a detailed guide.

## When to Use (Decision Table)

| Interview Signal | Topic to Know |
|-----------------|---------------|
| "How does attention work?" | Transformer_Architecture.md |
| "Explain GPT vs BERT" | Transformer_Architecture.md |
| "How would you adapt this model to your domain?" | Fine_Tuning.md |
| "We need to reduce training costs" | LLM_Training.md (scaling laws), Fine_Tuning.md (LoRA/QLoRA) |
| "How to give the model access to company docs?" | RAG.md |
| "Build a system that can use tools" | Agents_and_Tool_Use.md |
| "How do you measure if the model is better?" | Evaluation.md |
| "What if the model hallucinates?" | Evaluation.md, Safety_and_Alignment.md |
| "How do you make models safer?" | Safety_and_Alignment.md |
| "Explain the RLHF training process" | LLM_Training.md, Safety_and_Alignment.md |

## Files in this Section

- **Transformer_Architecture.md**: Self-attention, multi-head attention, positional encoding, architectural variants, inference optimizations (FlashAttention, KV cache)
- **LLM_Training.md**: Pre-training objectives, fine-tuning approaches, RLHF, DPO, Constitutional AI, scaling laws
- **Fine_Tuning.md**: Full fine-tuning vs PEFT methods (LoRA, QLoRA, adapters), quantization, when to fine-tune vs other approaches
- **RAG.md**: Retrieval-augmented generation pipeline, chunking, embeddings, vector DBs, evaluation, advanced techniques
- **Agents_and_Tool_Use.md**: AI agents, function calling, planning (ReAct, CoT), multi-agent systems, memory
- **Evaluation.md**: Benchmarks, LLM-as-a-Judge, hallucination detection, task-specific metrics, A/B testing
- **Safety_and_Alignment.md**: Alignment problem, RLHF, Constitutional AI, red-teaming, jailbreaks, bias, responsible deployment

## Top 15 LLM Interview Questions

1. **Explain self-attention and why it's better than RNNs for language modeling.**
   - See Transformer_Architecture.md

2. **What's the difference between BERT and GPT architectures?**
   - See Transformer_Architecture.md (encoder-decoder vs decoder-only)

3. **Why do we scale attention scores by √d_k?**
   - See Transformer_Architecture.md

4. **What is RLHF and how does it work?**
   - See LLM_Training.md and Safety_and_Alignment.md

5. **Explain LoRA. When would you use it over full fine-tuning?**
   - See Fine_Tuning.md

6. **What's the difference between fine-tuning and RAG? When would you choose each?**
   - See Fine_Tuning.md and RAG.md

7. **How do you evaluate an LLM when there's no ground truth?**
   - See Evaluation.md (LLM-as-a-Judge, human eval)

8. **What are hallucinations and how do you detect/mitigate them?**
   - See Evaluation.md and RAG.md

9. **Explain the RAG pipeline. What are common failure modes?**
   - See RAG.md

10. **What is DPO and how is it different from RLHF?**
    - See LLM_Training.md

11. **How would you build an agent that can use external tools?**
    - See Agents_and_Tool_Use.md

12. **What are scaling laws? What do they tell us about training LLMs?**
    - See LLM_Training.md

13. **What is FlashAttention and why does it matter?**
    - See Transformer_Architecture.md

14. **How do you prevent prompt injection attacks?**
    - See Safety_and_Alignment.md

15. **What is Constitutional AI?**
    - See LLM_Training.md and Safety_and_Alignment.md
