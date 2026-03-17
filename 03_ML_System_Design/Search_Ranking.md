# Search Ranking System

Design a search ranking system (like Google, Amazon product search, or internal enterprise search).

## 1. Problem Definition

### Clarifying Questions
- What are we searching? (web pages, products, documents)
- Scale? (queries per second, index size)
- Personalized or generic?
- Latency budget? (typically <200ms for web search)

### Scope
- Focus: query → ranked results
- In scope: query understanding, retrieval, ranking
- Out of scope: indexing pipeline details, query suggestions (unless asked)

### ML Task
- Ranking problem (order results by relevance)
- Also involves retrieval (find candidates) and query understanding

## 2. Metrics

### Offline Metrics
- **NDCG (Normalized Discounted Cumulative Gain)**: Standard ranking metric
- **MRR (Mean Reciprocal Rank)**: Where is the first relevant result?
- **Precision@k / Recall@k**: Quality of top results
- **Human relevance judgments**: Expert raters score query-document pairs

### Online Metrics
- **CTR (Click-Through Rate)**: % of queries with a click
- **Time to first click**: How fast do users find what they want?
- **Success rate**: % of queries with a satisfying click (dwell time > threshold)
- **Abandonment rate**: % of queries with no clicks
- **Reformulation rate**: % of users who modify their query

### Guardrail Metrics
- Latency (p99 < 200ms)
- Index coverage (% of corpus searchable)
- Zero-result rate (bad user experience)
- Diversity of results (avoid all results from one source)

## 3. Data

### Data Sources
**Query logs:**
- User search queries
- Clicked results (implicit positive labels)
- Dwell time (time spent on clicked page)
- Skip behavior (ignored results)

**Document corpus:**
- Text content (title, body, metadata)
- Structured data (category, price, ratings for products)
- Freshness signals (publication date, update time)

**User context:**
- Location
- Device type
- Search history
- Previous purchases (for e-commerce)

### Labeling
**Implicit labels:**
- Clicks are positive signals (but noisy due to position bias)
- Long dwell time = satisfied
- Immediate back = not satisfied

**Explicit labels:**
- Human raters grade query-document pairs (0-4 scale: irrelevant to perfect)
- Expensive but high quality
- Used for model evaluation and training

**Handling position bias:**
- Top results get more clicks regardless of relevance
- Use inverse propensity weighting
- Randomized experiments to collect unbiased data

### Features

**Query features:**
- Query length, query type (navigational, informational, transactional)
- Query embeddings (BERT, Sentence-BERT)
- Historical popularity of query

**Document features:**
- TF-IDF scores
- PageRank / authority score
- Freshness (recent documents may rank higher for news)
- Popularity (click count, rating)
- Document embeddings

**Query-document features:**
- BM25 score (best traditional IR metric)
- Semantic similarity (query embedding · document embedding)
- Exact match vs partial match
- Matched term positions (title, first paragraph, etc.)

**Context features:**
- User location (for local search)
- Time of day
- Personalization signals (past clicks, interests)

## 4. Model

### Multi-Stage Architecture

**Stage 1: Query Understanding**
- Spell correction (edit distance, language model)
- Query expansion (add synonyms, related terms)
- Intent classification (navigational, informational, transactional)
- Entity recognition (identify products, people, locations)

**Stage 2: Candidate Retrieval**
Goal: Reduce from millions/billions to ~10K candidates.

**Inverted index (traditional IR):**
- Map terms to documents containing them
- Boolean retrieval (AND, OR, NOT)
- BM25 scoring for quick filtering

**Embedding-based retrieval (neural IR):**
- Encode query and documents into dense vectors
- Approximate nearest neighbor search (FAISS, ScaNN)
- Two-tower model: separate encoders for query and document
- Handles semantic similarity (not just keyword match)

**Hybrid approach:**
- Combine inverted index and embeddings
- Get top-k from each, merge

**Stage 3: Ranking**
Goal: Order the 10K candidates by relevance.

**Model choice:**
- **GBDT (LightGBM, XGBoost)**: Strong baseline, handles 100s of features
- **Learning-to-Rank neural network**: Pointwise (predict relevance score), pairwise (predict which is better), or listwise (optimize NDCG directly)
- **Transformer-based cross-encoder**: BERT takes [query, document] and outputs relevance. Expensive but accurate.

**Features:**
- All features from section 3 (BM25, embeddings, popularity, etc.)
- Cross features (query length × document length)
- Historical CTR for this query-document pair

**Training:**
- Loss: Pairwise ranking loss or listwise softmax
- Labels: Human judgments + implicit clicks
- Time-based train/test split

**Stage 4: Re-ranking**
Goal: Apply business logic and personalization.

- Diversity (avoid showing 10 results from same domain)
- Freshness boost (for news queries)
- Personalization (boost results matching user interests)
- Deduplication (remove near-duplicates)

### Handling Cold Start
- New documents: Use content-based features (BM25, text embeddings)
- New queries: Fall back to traditional IR (inverted index)
- Gradually collect interaction data

## 5. Serving

### Architecture
```
User query
  → Query understanding (spell check, expansion)
  → Candidate retrieval (inverted index + ANN search)
  → Ranking (ML model scores candidates)
  → Re-ranking (business logic)
  → Return top 10 results
```

### Latency Budget
- Total: 200ms
- Retrieval: 50ms (index lookup + ANN search)
- Ranking: 100ms (score 10K candidates)
- Re-ranking: 30ms
- Overhead: 20ms

### Optimization Strategies

**Caching:**
- Cache popular queries (head queries account for huge traffic)
- Cache document embeddings (recompute only when document changes)

**Model optimization:**
- Quantization for neural models
- Distillation (compress large BERT into smaller model)
- Early exit (don't score all 10K, prune low-scoring candidates)

**Distributed serving:**
- Sharded index (split corpus across machines)
- Replicated model serving (load balance across GPUs)

**Batch processing:**
- Precompute document embeddings offline
- Update index daily (or hourly for fresh content)

### A/B Testing
- Randomize by user or query
- Compare CTR, success rate, time to first click
- Run for 1-2 weeks to account for day-of-week effects
- Watch for novelty effect (new ranking feels fresh but effect fades)

## 6. Monitoring

### Data Monitoring
- **Query distribution shift**: Are users searching for new topics?
- **Document corpus changes**: New documents added, old ones removed
- **CTR patterns**: Sudden drop may indicate ranking issue

### Model Monitoring
- **Ranking quality**: Track NDCG on live queries with human labels
- **Position bias**: Are we overfitting to top positions?
- **Zero-result queries**: Increasing rate means index or model issue

### Performance Monitoring
- **Latency**: p99 latency, broken down by stage
- **Error rates**: Failed queries, timeout errors
- **Index freshness**: Time lag between document update and index update

### Feedback Loops
**Risk:**
- Popular documents get more clicks, rank higher, get even more clicks
- Creates filter bubble, reduces diversity

**Mitigation:**
- Exploration: Randomly inject less-popular results
- Diversity penalty in re-ranking
- Periodically retrain on randomized serving data

### Retraining Strategy
- **Frequency**: Weekly or daily (depending on data volume)
- **Incremental updates**: For fast-moving content (news), update hourly
- **Online learning**: Update model weights in real-time (complex, can be unstable)
- **Shadow mode**: Run new model, compare predictions to old model before switching

## Key Trade-offs

**Precision vs Recall:**
- Retrieval stage: optimize recall (don't miss relevant documents)
- Ranking stage: optimize precision (top results must be good)

**Personalization vs Privacy:**
- More personalization = better results but raises privacy concerns
- Balance with anonymization and user control

**Freshness vs Quality:**
- Boost recent documents for news queries
- Prefer authoritative documents for evergreen queries

**Complexity vs Latency:**
- BERT cross-encoder is most accurate but slow
- Two-tower embedding model is faster but less accurate
- Use fast model for retrieval, accurate model for ranking

## Common Interview Follow-ups

**"How would you personalize search?"**
- User features: search history, location, past clicks
- Hybrid approach: generic ranking + personalization layer
- Cold start: fall back to non-personalized results

**"How would you handle multiple languages?"**
- Multilingual embeddings (mBERT, LaBSE)
- Language-specific inverted indices
- Translation (query → English, search English corpus)

**"How would you handle adversarial manipulation?"**
- SEO spam: Use quality signals (PageRank, domain authority)
- Click fraud: Filter bot traffic
- Regular audits with human raters

**"How would you do voice search?"**
- Speech-to-text preprocessing
- Longer, more conversational queries
- Different intent distribution (more informational, less navigational)
