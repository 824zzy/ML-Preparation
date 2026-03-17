# RAG (Retrieval-Augmented Generation)

## RAG Pipeline

**Three stages:**

1. **Retrieve:** Find relevant documents from knowledge base
2. **Augment:** Add retrieved context to prompt
3. **Generate:** LLM generates response using context

**Basic flow:**
```
Query → Embed → Vector search → Top-k docs → Format → LLM → Response
```

**When to use RAG:**
- Model needs external/private knowledge
- Knowledge changes frequently
- Need attribution/citations
- Reducing hallucinations on factual tasks

**When NOT to use RAG:**
- Task needs reasoning, not facts
- Knowledge is already in model
- Latency critical (RAG adds retrieval overhead)

## Chunking Strategies

Breaking documents into chunks for embedding and retrieval.

### Fixed-Size Chunking
- Split every N tokens (e.g., 512)
- Simple, fast
- Can split mid-sentence or mid-paragraph
- Often adds overlap (e.g., 512 tokens with 50 token overlap)

**Use when:** Documents are uniform, simplicity matters.

### Semantic Chunking
- Split at natural boundaries (paragraphs, sections)
- Preserves context
- Use embeddings to find topic boundaries

**Process:**
1. Split at paragraph boundaries
2. Embed each paragraph
3. Merge consecutive paragraphs with similar embeddings

**Use when:** Documents have clear structure (articles, books).

### Recursive Chunking
- Try to split at largest separator first (sections), then smaller (paragraphs), then sentences
- Keep chunks under max size
- Maintains hierarchy

**Example (LangChain):**
```
separators = ["\n\n", "\n", ". ", " "]
```

**Use when:** Documents have hierarchical structure.

### Sentence-Window Retrieval
- Embed individual sentences
- When sentence is retrieved, return surrounding context (window)
- Better relevance (small chunks) + better generation (large context)

**Use when:** You want precise retrieval but need context for generation.

**Practical advice:**
- Start with 512-1024 token chunks with 10-20% overlap
- Use semantic/recursive chunking if quality matters more than speed
- Smaller chunks = better retrieval precision, worse context
- Larger chunks = worse retrieval precision, better context

## Embedding Models and Vector Databases

### Embedding Models

**Popular choices:**
- OpenAI text-embedding-3-small/large (API-based)
- Sentence-Transformers (open source)
  - all-MiniLM-L6-v2 (fast, lightweight)
  - all-mpnet-base-v2 (better quality)
- E5 models (Microsoft, multilingual)
- Instructor embeddings (task-specific)
- BGE models (BAAI, SOTA on MTEB)

**Key properties:**
- Dimension: 384-1536 (higher = more expressive, slower)
- Max sequence length: 512-8192 tokens
- Domain: general vs specialized (code, scientific)

**Comparing embeddings:**
- MTEB benchmark (Massive Text Embedding Benchmark)
- Test on your domain (general benchmarks can mislead)

### Vector Databases

**Purpose:** Efficient similarity search over millions/billions of vectors.

**Popular options:**
- **Pinecone:** Managed, easy to use, scales well
- **Weaviate:** Open source, rich querying
- **Chroma:** Lightweight, local-first
- **Milvus:** Open source, high performance
- **FAISS:** Library (not full DB), very fast, CPU/GPU
- **Qdrant:** Open source, Rust-based, fast

**Key features:**
- Approximate nearest neighbor (ANN) search
- Filtering (metadata-based)
- Hybrid search (vector + keyword)
- Scalability (sharding, replication)

**Indexing methods:**
- HNSW (Hierarchical Navigable Small World): fast, memory-intensive
- IVF (Inverted File): good balance
- Product Quantization: compress vectors, trade accuracy for space

**Choosing a vector DB:**
- Starting out: Chroma (local), Pinecone (managed)
- Production scale: Pinecone, Weaviate, Qdrant
- Max performance: FAISS (requires more setup)

## Retrieval Methods

### Dense Retrieval
- Embed query and documents
- Find nearest neighbors by cosine similarity
- Standard approach, works well

**Pros:** Semantic matching, handles synonyms.
**Cons:** Can miss exact keyword matches.

### Sparse Retrieval (BM25)
- Traditional keyword-based search
- BM25: TF-IDF with length normalization

**Formula:**
```
score(D, Q) = Σ IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * |D| / avgdl))
```

**Pros:** Fast, good for exact keyword matches.
**Cons:** No semantic understanding.

### Hybrid Retrieval
- Combine dense + sparse
- Typically: weighted sum of scores or reciprocal rank fusion

**Reciprocal Rank Fusion (RRF):**
```
score(doc) = Σ 1 / (k + rank_i(doc))
```

Sum over retrieval methods, k = 60 typically.

**Why hybrid?**
- Dense: handles semantic similarity
- Sparse: handles exact terms, entity names
- Together: best of both

**When to use:**
- Queries with specific entities or technical terms
- When you see dense retrieval missing obvious matches

### Query Expansion
- Rewrite query to improve retrieval
- Hypothetical Document Embeddings (HyDE): generate hypothetical answer, embed that
- Multi-query: generate multiple variations of query, retrieve for each

## Reranking

**Problem:** Initial retrieval is fast but imprecise (top 100 docs may not be best 5).

**Solution:** Use more expensive model to rerank top-k results.

**Process:**
1. Retrieve top-k docs (e.g., k=100) with fast method
2. Score each doc with cross-encoder
3. Return top-n (e.g., n=5)

**Cross-encoder:**
- BERT-style model that encodes [query, doc] pair together
- More accurate than bi-encoder (separate query/doc encodings)
- Too slow for initial retrieval (can't pre-compute)

**Popular rerankers:**
- Cohere Rerank API
- bge-reranker models
- cross-encoder/ms-marco models

**When to use:**
- Large document corpus (>10k docs)
- Quality is critical
- You can afford the latency

## Evaluation of RAG Systems

RAG evaluation has two parts: retrieval quality and generation quality.

### Retrieval Metrics

**Hit Rate / Recall@k:**
- % of queries where relevant doc appears in top-k
- Measures if you retrieved the right docs

**Mean Reciprocal Rank (MRR):**
- Average of 1/rank of first relevant doc
- Rewards ranking relevant docs higher

**Precision@k:**
- % of top-k docs that are relevant

**NDCG (Normalized Discounted Cumulative Gain):**
- Rewards relevant docs at top positions
- Handles graded relevance (not just binary)

### Generation Metrics

**Faithfulness:**
- Does the answer match the retrieved context?
- Prevents hallucination beyond provided docs

**Answer Relevance:**
- Does the answer address the query?
- Can be high even if factually wrong

**Context Recall:**
- Is the ground truth answer present in retrieved context?
- Measures retrieval effectiveness

**Context Precision:**
- Are retrieved docs relevant?
- Fewer irrelevant docs = higher precision

### RAGAs Framework

Automated evaluation using LLM-as-a-judge:

1. **Faithfulness:** LLM checks if answer is supported by context
2. **Answer Relevance:** LLM scores how well answer addresses query
3. **Context Recall:** LLM checks if context contains ground truth
4. **Context Precision:** LLM identifies which retrieved chunks were useful

**Advantages:**
- No human labeling needed
- Scales to large test sets

**Disadvantages:**
- Requires LLM API calls (cost)
- LLM judge can be wrong

### End-to-End Evaluation

**Human evaluation:**
- Gold standard but expensive
- Sample random queries, have experts rate responses

**LLM-as-a-judge:**
- Use strong LLM (GPT-4) to rate answers
- Give rubric and examples
- Often correlates well with human judgment

**A/B testing:**
- Deploy two versions, measure user satisfaction
- Click-through rate, time on page, thumbs up/down

## Advanced RAG Techniques

### Query Rewriting

**Problem:** User queries are often poorly phrased for retrieval.

**Approaches:**
1. **Prompt LLM to rewrite:** "Rephrase this query for search: {query}"
2. **HyDE:** Generate hypothetical answer, embed and search with that
3. **Multi-query:** Generate multiple query variations, retrieve for each

**Example:**
- User: "Why is my bill high?"
- Rewritten: "What factors contribute to increased billing costs?"

### Multi-Hop Reasoning

**Problem:** Single retrieval isn't enough for complex questions.

**Approach:**
1. Retrieve docs for initial query
2. LLM identifies what's missing
3. Generate follow-up query
4. Retrieve again
5. Combine contexts for final answer

**Example:**
- Q: "Did the winner of the 2020 election previously hold office in Chicago?"
- Hop 1: Retrieve "Who won 2020 election?" → Joe Biden
- Hop 2: Retrieve "Did Joe Biden hold office in Chicago?" → No

### Self-RAG

Model decides when to retrieve and what to trust:
1. Generate initial response
2. Model judges if it needs retrieval
3. If yes, retrieve and regenerate
4. Model critiques its own outputs for factuality

**Advantages:**
- Adaptive (only retrieves when needed)
- More reliable (self-verification)

**Disadvantages:**
- Requires specially trained model

### Graph RAG

**Problem:** Documents have relationships that vector search misses.

**Approach:**
1. Build knowledge graph from documents (entities + relations)
2. Retrieve subgraphs relevant to query
3. Use graph structure for context

**When to use:** Multi-hop reasoning, when relationships matter (legal documents, scientific papers).

## RAG vs Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| Knowledge update | Easy (add to index) | Hard (retrain) |
| Setup cost | Low (embed + index) | High (training) |
| Inference cost | High (retrieval + LLM) | Low (just LLM) |
| Latency | Higher | Lower |
| Attribution | Yes (cite sources) | No |
| Task adaptation | Limited | Full |
| Use case | External knowledge | Behavior/style change |

**Combining both:**
- Fine-tune for task/style
- RAG for knowledge
- Example: Fine-tuned medical LLM + RAG over latest research

## Common Failure Modes

1. **Retrieval failure:** Relevant docs not retrieved
   - Fix: Improve embeddings, try hybrid search, tune chunk size

2. **Irrelevant context:** Retrieved docs don't help
   - Fix: Better chunking, reranking, filter by metadata

3. **Lost in the middle:** LLM ignores context in middle of long contexts
   - Fix: Rerank so best docs are at start/end, or summarize chunks

4. **Conflicting information:** Docs contradict each other
   - Fix: Add timestamps, prioritize sources, let LLM reconcile

5. **Hallucination despite context:** LLM ignores retrieved docs
   - Fix: Better prompting ("answer ONLY using context"), instruct tuning

6. **Over-reliance on retrieval:** Answer in docs but LLM can't find it
   - Fix: Highlight key passages, summarize docs first, better prompts
