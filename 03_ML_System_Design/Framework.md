# ML System Design Framework

This is the universal template. Apply it to every case study.

## 1. Problem Definition

### Clarify Requirements
- What is the business goal? (revenue, engagement, safety)
- Who are the users?
- What is the scale? (DAU, QPS, data volume)
- What are the latency requirements?
- What existing systems/data do we have?

### Define Scope
- What are we building? (new feature, improving existing system)
- What is in scope vs out of scope?
- Are there regulatory constraints? (GDPR, bias/fairness)

### Identify ML Task Type
- Classification (binary, multi-class, multi-label)
- Regression
- Ranking
- Generation
- Clustering/segmentation
- Retrieval

## 2. Metrics

### Offline Metrics
These evaluate model quality on held-out data.

**Classification:**
- Precision, recall, F1
- AUC-ROC, AUC-PR
- Confusion matrix analysis

**Ranking:**
- NDCG, MAP, MRR
- Precision@k, Recall@k

**Regression:**
- MSE, RMSE, MAE
- R², MAPE

**Generation:**
- BLEU, ROUGE (text)
- Human eval scores

### Online Metrics
These measure real business impact.

**User Engagement:**
- CTR (click-through rate)
- Conversion rate
- Time spent
- Return rate

**Business:**
- Revenue per user
- Ad revenue
- Subscription retention

**Platform Health:**
- Query success rate
- User satisfaction scores
- Reported content ratio (for moderation)

### Guardrail Metrics
These ensure you don't break things.

- Latency (p50, p99)
- Error rate
- System load
- Fairness metrics (demographic parity, equalized odds)
- User complaints/reports

### Key Point
Offline metrics are necessary but not sufficient. A model with better AUC can have worse online CTR due to position bias, user fatigue, or novelty effects.

## 3. Data

### Data Sources
- Where does the data come from? (user logs, transactions, sensors)
- What is the data volume and growth rate?
- What is the data freshness? (streaming, daily batch)

### Labeling
**Explicit labels:**
- Manual annotation (expensive, high quality)
- Crowdsourcing (scalable, needs quality control)
- Expert labeling (specialized domains)

**Implicit labels:**
- User interactions (clicks, likes, time spent)
- Positive labels are easy, negatives are noisy

**Weak supervision:**
- Heuristics and rules
- Semi-supervised learning
- Distillation from larger models

### Feature Engineering

**User features:**
- Demographics (age, location, language)
- Historical behavior (past clicks, purchases)
- Session context (device, time of day)

**Item features:**
- Content features (title, description, image)
- Metadata (category, price, author)
- Popularity signals (view count, rating)

**Context features:**
- Query (for search/ads)
- Current page (for recommendations)
- Temporal (day of week, season)

**Interaction features:**
- User-item history
- Co-occurrence patterns
- Embeddings from collaborative filtering

### Data Pipeline
- ETL process (extract, transform, load)
- Feature store (avoid train-serve skew)
- Data validation (schema checks, distribution checks)

## 4. Model

### Multi-Stage Architecture
Most production systems use a funnel approach:

**Stage 1: Candidate Generation (Recall)**
- Goal: Narrow down from millions to thousands
- Speed matters more than precision
- Methods: collaborative filtering, embedding similarity, inverted index

**Stage 2: Ranking (Precision)**
- Goal: Order candidates by relevance/quality
- Can use heavier models
- Methods: learning-to-rank, pointwise/pairwise/listwise loss

**Stage 3: Re-ranking (Business Logic)**
- Goal: Apply business rules, diversity, fairness
- Methods: MMR (maximal marginal relevance), personalization, ads auction

### Model Selection

**Simple Baselines:**
- Logistic regression (interpretable, fast, strong baseline)
- GBDT (XGBoost, LightGBM) - handles features well, no need for scaling

**Neural Networks:**
- Wide & Deep (memorization + generalization)
- Two-tower models (user encoder + item encoder)
- Transformer-based (for text/sequential data)

**Trade-offs:**
- Complexity vs latency
- Accuracy vs interpretability
- Cold start handling

### Training Strategy

**Data split:**
- Time-based split (more realistic than random)
- Watch for data leakage

**Loss function:**
- Pointwise (binary cross-entropy)
- Pairwise (ranking loss)
- Listwise (softmax over list)

**Handling imbalance:**
- Resampling (undersample negatives)
- Class weights
- Focal loss

**Regularization:**
- L2 for linear models
- Dropout for neural networks
- Early stopping

**Distributed training:**
- Data parallelism for large datasets
- Model parallelism for large models

## 5. Serving

### Batch vs Real-Time

**Batch Predictions:**
- Pre-compute predictions for all users/items
- Store in cache (Redis, Memcached)
- Good for: daily recommendations, email digests
- Latency: N/A (precomputed)

**Real-Time Predictions:**
- Compute on-demand per request
- Good for: search ranking, ads, fraud detection
- Latency: <100ms typical target

**Hybrid:**
- Batch candidate generation + real-time ranking
- Most common approach for large-scale systems

### Latency Optimization

**Model optimization:**
- Quantization (FP32 → INT8)
- Pruning
- Knowledge distillation (teacher-student)

**Infrastructure:**
- Model caching
- Feature caching
- GPU serving for heavy models
- Edge deployment for ultra-low latency

**Approximation:**
- ANN (approximate nearest neighbors) for retrieval
- Top-k pruning
- Early exit in neural networks

### A/B Testing
- Control vs treatment groups
- Randomization unit (user, session)
- Sufficient sample size
- Statistical significance (p-value < 0.05 typical)
- Beware of network effects

## 6. Monitoring

### Data Monitoring

**Data drift:**
- Feature distributions change over time
- Detect with KL divergence, chi-squared test
- Example: sudden spike in mobile traffic

**Label drift:**
- Ground truth distribution changes
- Example: new fraud patterns emerge

**Training-serving skew:**
- Features computed differently in training vs serving
- Use feature store to prevent this

### Model Monitoring

**Performance metrics:**
- Track offline metrics on live data
- Track online metrics from A/B tests
- Set up alerts for drops

**Prediction distribution:**
- Are predictions well-calibrated?
- Are we seeing unexpected patterns?

**Latency and throughput:**
- p50, p99 latency
- QPS (queries per second)
- Error rates

### Feedback Loops

**Positive feedback loops:**
- Recommended content gets more clicks, gets recommended more
- Can create filter bubbles

**Negative feedback loops:**
- Showing low-quality content reduces engagement, reduces training data

**Mitigation:**
- Exploration (epsilon-greedy, Thompson sampling)
- Randomization in serving
- Diversification

### Retraining Strategy

**When to retrain:**
- Scheduled (daily, weekly)
- Triggered (when drift detected)
- Online learning (continuous updates)

**What to retrain:**
- Full retraining vs incremental updates
- Cost/benefit trade-off

**Shadow deployment:**
- Run new model alongside old model
- Compare predictions before switching
