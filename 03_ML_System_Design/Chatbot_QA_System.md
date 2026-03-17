# Chatbot / QA System (LLM-based)

Design a question-answering chatbot for a company's internal knowledge base or customer support (e.g., Intercom, Zendesk, or internal wiki assistant).

## 1. Problem Definition

### Clarifying Questions
- What domain? (customer support, internal docs, general QA)
- Scale? (queries per day, size of knowledge base)
- Requirements? (accuracy, latency, cost per query)
- User expectations? (conversational vs direct answers)
- Existing knowledge base format? (structured FAQ, unstructured docs, tickets)

### Scope
- Focus: Retrieve relevant context + generate answer with LLM
- In scope: RAG pipeline, guardrails, fallback to human
- Out of scope: Building the knowledge base itself (assume it exists)

### Modern Context
This is an LLM-era problem. Traditional IR + rule-based QA is obsolete. The standard approach now is RAG (Retrieval-Augmented Generation).

## 2. Metrics

### Offline Metrics
- **Answer correctness:** Human eval on test set (binary: correct/incorrect)
- **Answer relevance:** Does answer address the question? (1-5 scale)
- **Groundedness:** Is answer supported by retrieved context? (hallucination check)
- **Retrieval quality:** Recall@k (are relevant docs in top-k?)
- **Latency:** End-to-end response time

### Online Metrics
- **User satisfaction:** Thumbs up/down, CSAT score
- **Resolution rate:** % of queries resolved without human escalation
- **Escalation rate:** % of queries sent to human agent
- **Follow-up questions:** If user asks clarification, answer was probably bad
- **Session abandonment:** User gives up mid-conversation

### Guardrail Metrics
- **Hallucination rate:** % of answers not grounded in retrieved docs
- **Harmful content rate:** % of answers with toxic/biased content
- **Off-topic rate:** % of answers unrelated to question
- **Latency:** p99 < 3s (users tolerate slower LLM responses than search)
- **Cost per query:** LLM inference is expensive (track token usage)

## 3. Data

### Knowledge Base
**Structured data:**
- FAQs (question + answer pairs)
- Product docs (Markdown, HTML)
- Policy documents (PDFs)

**Unstructured data:**
- Past support tickets (solved conversations)
- Internal wiki pages
- Slack messages (if internal chatbot)

**Metadata:**
- Document freshness (when was it last updated?)
- Source authority (official doc vs random wiki page)
- Access control (some docs are confidential)

### Preprocessing
- Chunk documents (LLMs have context limits, split long docs)
- Typical chunk size: 500-1000 tokens with overlap
- Preserve structure (keep headers with their sections)
- Remove boilerplate (footers, navigation)

### Labeling for Evaluation
**Human-curated test set:**
- Common questions + expected answers
- Edge cases (ambiguous questions, questions with no answer)

**Golden retrieval set:**
- For each question, which docs should be retrieved?
- Measure retrieval quality

**Hallucination detection:**
- Sample generated answers, check if grounded in context

### Features for Ranking Retrieved Docs
- BM25 score (traditional IR)
- Embedding similarity (dense retrieval)
- Recency (prefer recent docs for time-sensitive questions)
- Authority (official docs over user-contributed)
- Past usage (docs that helped previous users)

## 4. Model

### RAG (Retrieval-Augmented Generation) Pipeline

**Step 1: Query Understanding**
- Clarify ambiguous queries (if too short, ask follow-up)
- Detect intent (factual question, troubleshooting, complaint)
- Extract entities (product names, error codes)

**Step 2: Retrieval**
Goal: Find top-k most relevant documents (typically k=3-10).

**Traditional IR (Sparse Retrieval):**
- BM25 on inverted index (fast, keyword-based)
- Good for queries with specific terms (product names, error codes)

**Dense Retrieval (Embedding-based):**
- Encode query and documents into dense vectors
- Compute cosine similarity
- Use FAISS or similar for fast ANN search
- Models: Sentence-BERT, Contriever, E5, OpenAI embeddings

**Hybrid Retrieval:**
- Combine BM25 + dense retrieval (best of both worlds)
- Reciprocal Rank Fusion or learned weights

**Re-ranking:**
- Stage 1: Retrieve top-100 with fast models
- Stage 2: Re-rank with cross-encoder (BERT takes [query, doc] as input)
- Return top-k for LLM context

**Step 3: Context Construction**
- Concatenate retrieved docs into prompt
- Add metadata (source, date)
- Truncate if exceeds LLM context window

**Step 4: Answer Generation**
Prompt template:
```
You are a helpful assistant. Answer the user's question based on the provided context.

Context:
{retrieved_doc_1}
{retrieved_doc_2}
...

Question: {user_query}

Instructions:
- Answer based on the context above
- If the context doesn't contain the answer, say "I don't have enough information"
- Cite sources when possible
```

**Step 5: Post-processing**
- Citation generation (link to source docs)
- Hallucination check (does answer match retrieved context?)
- Safety check (toxic content filter)

### LLM Selection

**Hosted APIs:**
- GPT-4 (OpenAI): High quality, expensive ($0.03/1K tokens)
- GPT-3.5: Cheaper ($0.002/1K tokens), slightly lower quality
- Claude (Anthropic): Strong safety, good for sensitive domains

**Open-source models:**
- Llama 3 (70B): Strong performance, self-hosted
- Mixtral (8x7B): Good quality-to-cost ratio
- Fine-tuned smaller models (7B-13B): Cheaper, domain-specific

**Trade-offs:**
- Quality vs cost (GPT-4 is best but 15x more expensive than GPT-3.5)
- Latency vs accuracy (larger models are slower)
- Self-hosted vs API (control vs convenience)

### Fine-Tuning Strategy
If using open-source model, fine-tune on domain data:
- Curate (question, context, answer) triples
- Fine-tune with supervised learning (next-token prediction)
- Use RLHF or DPO for preference alignment

### Guardrails

**Input guardrails:**
- Detect jailbreak attempts (prompt injection)
- Filter profanity, harmful requests
- Rate limiting (prevent abuse)

**Output guardrails:**
- Hallucination detection (fact-check against retrieved context)
- Toxicity filter (PerspectiveAPI, custom classifier)
- PII redaction (don't leak emails, SSNs)

**Response validation:**
- If answer is too generic ("I can help with that!"), regenerate
- If answer is off-topic, fall back to retrieval results only

## 5. Serving

### Architecture
```
User query
  → Query understanding (clarification, entity extraction)
  → Retrieval (BM25 + dense retrieval + re-ranking)
  → Prompt construction
  → LLM generation
  → Post-processing (citation, safety checks)
  → Return answer
```

### Latency Budget
- Total: <3s (users tolerate slower responses for chatbots)
- Retrieval: 200ms (BM25 + ANN + re-rank)
- LLM generation: 1-2s (depends on model size, output length)
- Post-processing: 100ms

### Optimization Strategies

**Caching:**
- Cache embeddings for all knowledge base docs (recompute only when docs change)
- Cache answers for common questions (FAQ)
- Cache LLM completions (deterministic queries)

**Batching:**
- If multiple users ask at once, batch LLM requests (GPU efficient)

**Streaming:**
- Stream LLM output token-by-token (feels faster, better UX)

**Model optimization:**
- Quantization (FP16, INT8) for self-hosted models
- Flash Attention (2x faster inference)
- Speculative decoding (draft with small model, verify with large model)

**Early termination:**
- If question is in FAQ, return cached answer (skip LLM)
- If retrieval finds exact match, return it directly

### Fallback Strategies

**When to fall back to human:**
- LLM says "I don't know"
- User gives negative feedback
- Query involves sensitive topics (legal, medical)
- Escalation keywords ("speak to a human")

**Graceful degradation:**
- If LLM is down, return retrieval results only (traditional search)
- If retrieval is slow, use cached FAQ

### A/B Testing
- Control: Existing system (rule-based or human-only)
- Treatment: RAG-based chatbot
- Metrics: Resolution rate, user satisfaction, cost per query
- Run for 2-4 weeks (need enough queries for significance)

## 6. Monitoring

### Data Monitoring

**Knowledge base drift:**
- Docs are added, updated, deprecated
- Re-index and re-embed regularly (daily or weekly)

**Query distribution shift:**
- New product launch = new types of questions
- Seasonal trends (e.g., holiday return policy questions)

**User behavior change:**
- Users learn how to phrase queries better
- Or learn to bypass chatbot and go straight to human

### Model Monitoring

**Retrieval quality:**
- Track Recall@k on golden retrieval set
- If recall drops, re-tune retrieval model

**Answer quality:**
- Human eval on sample of live queries (weekly)
- Track thumbs up/down ratio

**Hallucination rate:**
- Sample answers, check if grounded in retrieved context
- Use NLI model (natural language inference) to auto-detect hallucinations

**Latency:**
- Track p50, p99 latency by component (retrieval, LLM, total)
- Alert if p99 > 5s

### Safety Monitoring

**Harmful outputs:**
- Sample answers for toxicity, bias
- Use classifiers (PerspectiveAPI, custom)

**Jailbreak attempts:**
- Log and analyze failed guardrail checks
- Update filters based on new attack patterns

**PII leakage:**
- Check if answers contain emails, phone numbers, SSNs
- Redact automatically

### Business Monitoring

**Cost per query:**
- LLM tokens used (input + output)
- Retrieval compute cost
- Track over time, optimize if too high

**Resolution rate:**
- % of queries handled without human escalation
- Goal: >70% (but depends on domain complexity)

**Human workload:**
- Are humans spending less time on repetitive questions?
- Or are they fixing chatbot mistakes (worse than before)?

### Feedback Loops

**Risk:**
- Chatbot gives wrong answer, user believes it, user makes mistake
- Creates negative feedback (users trust chatbot less)

**Mitigation:**
- Clear disclaimers ("This is AI-generated, verify important info")
- Easy escalation to human
- Track and fix recurring errors quickly

**Positive loop:**
- Good answers get upvotes, added to training data
- Model improves over time

## Key Trade-offs

**Retrieval Quality vs Latency:**
- Dense retrieval + re-ranking is accurate but slow
- BM25 is fast but misses semantic matches
- Solution: Hybrid (BM25 for top-100, re-rank top-10)

**LLM Size vs Cost/Latency:**
- GPT-4 is best but expensive and slow
- GPT-3.5 is 15x cheaper and faster but lower quality
- Solution: Use GPT-4 for complex queries, GPT-3.5 for simple ones

**Groundedness vs Fluency:**
- Strict grounding (only say what's in context) = safe but robotic
- Creative generation = fluent but may hallucinate
- Solution: Tune prompt and temperature (low temp = more grounded)

**Automation vs Human Handoff:**
- Full automation = scalable but error-prone
- Human in loop = high quality but expensive
- Solution: Automate easy queries, escalate hard ones

## Common Interview Follow-ups

**"How do you handle questions with no answer in the knowledge base?"**
- LLM should say "I don't have enough information" (train on such examples)
- Don't hallucinate
- Suggest related questions or escalate to human

**"How do you keep the knowledge base up to date?"**
- Automated ingestion from source systems (docs, tickets)
- Versioning (track when docs change)
- Re-index and re-embed on updates
- Monitor for stale docs (low usage, old timestamp)

**"How do you handle multi-turn conversations?"**
- Maintain conversation history (previous Q&A pairs)
- Coreference resolution ("What about pricing?" after asking about "Product X")
- Include recent context in LLM prompt
- Limit history to last N turns (avoid context overflow)

**"How do you handle ambiguous questions?"**
- Ask clarifying questions ("Which product are you asking about?")
- Provide multiple answers if multiple interpretations
- Use dialogue management (track conversation state)

**"How do you prevent prompt injection attacks?"**
- Input validation (detect adversarial patterns)
- Sandboxing (LLM can't execute code or access external systems)
- Output validation (check for leaked system prompts)

**"How do you evaluate open-ended text generation?"**
- Human eval (gold standard, expensive)
- LLM-as-judge (GPT-4 evaluates GPT-3.5 outputs)
- Automated metrics (BLEU, ROUGE, BERTScore) - useful for comparison but not absolute quality

**"What if the user asks a question outside the domain?"**
- Detect out-of-domain queries (classifier or keyword check)
- Politely decline ("I can only answer questions about X")
- Redirect to appropriate resource

**"How do you handle multiple languages?"**
- Multilingual embeddings (mBERT, LaBSE) for retrieval
- Multilingual LLMs (GPT-4, mT5)
- Or: Translate query to English, generate answer, translate back
