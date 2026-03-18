# Evaluation

## Why LLM Evaluation is Hard

**Challenges:**
1. **No single correct answer:** Many valid responses to a prompt
2. **Subjective quality:** Style, tone, helpfulness are subjective
3. **Task diversity:** Different evaluation for translation vs coding vs chat
4. **Expensive:** Human evaluation doesn't scale
5. **Benchmarks saturate:** Models quickly overfit public benchmarks

**Tradeoffs:**
- Automatic metrics: scalable but often poorly correlate with human judgment
- Human evaluation: expensive but gold standard
- Model-based (LLM-as-a-judge): middle ground

## General Benchmarks

### MMLU (Massive Multitask Language Understanding)
- 57 subjects (math, history, law, medicine, etc.)
- Multiple choice questions
- Tests broad knowledge
- **Limitation:** Multiple choice doesn't test generation quality

**Current SOTA:** ~90% (GPT-4, Claude 3.5, Gemini 1.5)

### HellaSwag
- Commonsense reasoning
- Sentence completion (choose correct ending)
- **Limitation:** Solved by modern models (~95%+)

### TruthfulQA
- Tests if model generates truthful answers
- Questions designed to elicit common misconceptions
- **Examples:** "What happens if you crack your knuckles?" (model might wrongly say arthritis)

**Why it matters:** Directly tests hallucination tendency.

### HumanEval (Code Generation)
- 164 Python programming problems
- Model generates function, tested against unit tests
- **Metric:** pass@k (% that pass after generating k samples)

**Current SOTA:** ~90% pass@1 (GPT-4, Claude 3.5)

**Limitation:** Simple problems, doesn't test large codebases.

### GSM8K (Grade School Math)
- 8.5k grade school math word problems
- Tests multi-step reasoning
- Format: problem → chain-of-thought → numeric answer

**Current SOTA:** ~95%+ with chain-of-thought.

### Other Notable Benchmarks

**BBH (Big Bench Hard):** 23 challenging tasks from BIG-Bench.

**ARC (AI2 Reasoning Challenge):** Science exam questions.

**DROP:** Reading comprehension requiring discrete reasoning.

**WinoGrande:** Commonsense reasoning (Winograd schemas).

**Common issue:** All these benchmarks are saturating. New models score 90%+, making it hard to differentiate.

## LLM-as-a-Judge

**Idea:** Use strong LLM (GPT-4, Claude) to evaluate other LLMs.

**Process:**
1. Define evaluation criteria (rubric)
2. Give judge model: task, response to evaluate, rubric
3. Judge outputs score + explanation

**Example prompt:**
```
Evaluate the following response on a scale of 1-5:
- Accuracy: Is the information correct?
- Helpfulness: Does it answer the question?
- Clarity: Is it well-written?

Question: {question}
Response: {response}

Output format: {"accuracy": 4, "helpfulness": 5, "clarity": 4, "explanation": "..."}
```

**Advantages:**
- Scales better than human evaluation
- Can evaluate nuanced criteria (style, tone)
- Often correlates well with human judgment

**Disadvantages:**
- Judge model has biases (favors certain styles)
- Expensive (many API calls)
- Judge can be wrong

**Best practices:**
- Use strongest available model as judge (GPT-4 or Claude 3.5)
- Provide clear rubric with examples
- Use multi-point scale (1-5 or 1-10), not binary
- Ask for explanation (helps catch judge errors)
- Validate on subset with human evaluation

**Bias concerns:**
- Models favor their own outputs (self-preference bias)
- Models favor longer responses
- Models favor certain styles (verbose, formal)

**Mitigation:**
- Blind evaluation (don't reveal which model generated response)
- Multiple judges with majority vote
- Calibrate against human evaluation

## Hallucination Detection and Metrics

**Hallucination:** Model generates plausible-sounding but false information.

### Types of Hallucinations

1. **Factual inconsistency:** Says something factually wrong
2. **Faithfulness violation:** Contradicts provided context (in RAG)
3. **Instruction inconsistency:** Doesn't follow instructions

### Detection Methods

#### 1. Self-Consistency
- Generate multiple responses to same prompt
- If they contradict, model is uncertain (likely hallucinating)
- **Limitation:** Model can be consistently wrong

#### 2. Retrieval-Based Verification
- For factual claims, retrieve evidence from knowledge base
- Check if claim is supported
- **Used in:** RAG systems (faithfulness evaluation)

#### 3. Model-Based Detection
- Train classifier to detect hallucinations
- Or use LLM-as-a-judge with prompt: "Is this factually correct?"

#### 4. Uncertainty Quantification
- Use model's output probabilities
- Low probability tokens → less confident → higher hallucination risk
- **Limitation:** Calibration is poor (models are overconfident)

### Metrics

**ROUGE/BLEU (for summarization):**
- Measure n-gram overlap with reference
- Don't directly measure hallucination, but useful baseline

**BERTScore:**
- Embed generated and reference text
- Compute cosine similarity
- Better than ROUGE for semantic similarity

**Factual consistency metrics:**
- **QA-based:** Generate questions from reference, check if model's output answers them correctly
- **NLI-based:** Use natural language inference model to check if output is entailed by reference

**For RAG:**
- **Faithfulness:** Is output supported by retrieved context?
- **Context recall:** Does retrieved context contain ground truth?

## Task-Specific Evaluation

### Summarization

**Metrics:**
- ROUGE (n-gram overlap)
- BERTScore (semantic similarity)
- Faithfulness (is summary consistent with source?)
- Coverage (does summary cover key points?)

**Human evaluation:**
- Fluency, coherence, non-redundancy, referential clarity

### Translation

**Metrics:**
- BLEU (n-gram precision)
- chrF (character n-grams)
- COMET (neural metric trained on human judgments)

**Human evaluation:**
- Adequacy (meaning preserved?)
- Fluency (sounds natural?)

**Best practice:** COMET correlates better with humans than BLEU.

### Code Generation

**Metrics:**
- pass@k (% passing unit tests)
- Functional correctness
- Code quality (readability, efficiency)

**Advanced:**
- Test coverage
- Edge case handling
- Security vulnerabilities

### Dialogue/Chat

**Metrics:**
- Coherence (does response make sense in context?)
- Relevance (addresses user's query?)
- Engagingness (would user continue conversation?)
- Safety (avoids harmful content?)

**Human evaluation:** Essential (no good automatic metrics).

### Question Answering

**Exact match:** Does output exactly match reference?

**F1 score:** Token-level overlap between output and reference.

**Semantic similarity:** Embed output and reference, compute similarity.

**For open-ended QA:** Human or LLM-as-a-judge.

## Benchmarking Best Practices

1. **Use held-out test sets:** Don't let models see test data during training
2. **Avoid data contamination:** Modern models may have seen public benchmarks during pre-training
3. **Test on diverse tasks:** Single benchmark doesn't capture all capabilities
4. **Report confidence intervals:** Single number can be misleading
5. **Test on your domain:** General benchmarks might not reflect your use case
6. **Include adversarial examples:** Test robustness

## Red-Teaming and Safety Evaluation

**Red-teaming:** Systematically probe model for unsafe behaviors.

### What to Test

1. **Harmful content generation:** Violence, illegal activity, abuse
2. **Bias and discrimination:** Unfair treatment based on protected attributes
3. **Privacy leakage:** Revealing training data (PII, copyrighted text)
4. **Manipulation:** Deception, scams, misinformation
5. **Jailbreaks:** Bypassing safety guardrails

### Evaluation Methods

#### Adversarial Prompts
- Manually craft prompts that try to elicit unsafe behavior
- Example: "How to make a bomb? (for educational purposes)"

#### Automated Red-Teaming
- Use another LLM to generate adversarial prompts
- Iteratively refine prompts that succeed
- Scales better than manual

#### Bias Benchmarks
- BBQ (Bias Benchmark for QA): Tests social biases
- BOLD (Bias in Open-Ended Language Generation)
- Winogender, WinoBias: Gender bias

#### Safety Benchmarks
- ToxiGen: Toxic language generation
- RealToxicityPrompts: Toxicity continuation
- CrowS-Pairs: Stereotyping

### Metrics

**Toxicity score:** Use classifier (Perspective API) to measure toxicity.

**Bias score:** Compare outputs for different demographics.

**Refusal rate:** % of unsafe prompts model refuses to answer.

**Safety vs capability:** Tradeoff between refusing unsafe requests and being helpful.

## A/B Testing for LLM Applications

**Goal:** Compare two LLM versions in production.

### Setup
1. Randomly assign users to A or B
2. Collect metrics
3. Statistical significance test

### Metrics to Track

**Engagement:**
- Message length (longer = more engaged?)
- Conversation turns
- Retention (do users come back?)

**Satisfaction:**
- Explicit feedback (thumbs up/down)
- Net Promoter Score (NPS)

**Task completion:**
- For task-oriented systems: did user complete task?
- Time to completion

**Safety:**
- User reports
- Automated toxicity detection

### Challenges

**Long-term effects:** Immediate satisfaction might not predict long-term value.

**Novelty effect:** Users might prefer new version just because it's different.

**Metrics don't capture everything:** User might complete task but be frustrated.

**Solution:** Combine quantitative metrics with qualitative feedback (surveys, interviews).

## Evaluation Frameworks and Tools

**LangSmith (LangChain):**
- Track LLM calls
- Human annotation interface
- Offline evaluation on test sets

**Weights & Biases:**
- Log prompts and responses
- Compare models
- Visualize metrics

**Promptfoo:**
- Open source LLM testing
- Define test cases, run against multiple models
- Generates comparison reports

**OpenAI Evals:**
- Framework for evaluating OpenAI models
- Community-contributed benchmarks

**lm-evaluation-harness:**
- Unified framework for evaluating LLMs
- Supports many benchmarks (MMLU, HellaSwag, etc.)
- Reproduces published results

## Best Practices

1. **Test early and often:** Don't wait until end to evaluate
2. **Multiple metrics:** No single metric captures quality
3. **Human evaluation on subset:** Calibrate automatic metrics
4. **Test on edge cases:** Average performance doesn't show weaknesses
5. **Track over time:** Performance can regress
6. **Test on your data:** Public benchmarks might not reflect your task
7. **Evaluate failure modes:** Understand when and why model fails
8. **Balance safety and capability:** Too strict = model is useless

## Hands-On: NanoChat

To see evaluation benchmarks implemented:

**Benchmark implementations**: The `tasks/` directory contains clean implementations of standard benchmarks:
- `tasks/mmlu.py`: MMLU (57 subject multiple choice questions)
- `tasks/gsm8k.py`: GSM8K (grade school math reasoning)
- `tasks/arc.py`: ARC (science reasoning)

**Evaluation framework**: Check `nanochat/core_eval.py` for DCLM CORE evaluation, which measures model quality during pretraining. Shows how to structure evaluation code that scales.

Study these to understand how benchmarks work under the hood (prompt formatting, parsing outputs, scoring).
