# Ads Click Prediction System

Design a system to predict ad click probability (like Google Ads, Facebook Ads, or LinkedIn Ads).

## 1. Problem Definition

### Clarifying Questions
- What platform? (search ads, display ads, feed ads)
- Scale? (impressions per day, number of advertisers)
- Auction mechanism? (second-price, first-price, VCG)
- Cold start? (new ads, new users)

### Scope
- Focus: Predict P(click | user, ad, context) for ad ranking
- In scope: CTR prediction model, auction, serving
- Out of scope: Ad creative generation, advertiser bidding strategies

### Business Context
- Advertisers bid for ad placements
- Platform ranks ads by `bid × pCTR × quality_score`
- Revenue = sum of winning bids (per click or per impression)
- Balance: maximize revenue while maintaining user experience

## 2. Metrics

### Offline Metrics
- **AUC (Area Under ROC Curve)**: Standard for binary classification
- **Log loss (cross-entropy)**: Penalizes confident wrong predictions
- **Calibration**: Do predicted probabilities match true click rates?
  - Plot predicted CTR vs observed CTR in buckets
  - Well-calibrated model: predicted 5% = observed 5%

### Online Metrics
- **CTR (Click-Through Rate)**: Overall % of impressions clicked
- **Revenue per 1000 impressions (RPM)**: Direct business metric
- **Advertiser ROI**: Conversions per ad spend (long-term health)
- **User engagement**: Time spent, bounce rate (don't ruin UX)

### Guardrail Metrics
- Ad load (% of content that is ads)
- Latency (auction + prediction < 100ms)
- Policy violations (blocked ads, user reports)
- Advertiser satisfaction (bid fulfillment rate)

### Key Point
A model with better AUC may hurt revenue if it's miscalibrated. Bid × pCTR requires calibrated probabilities, not just relative ranking.

## 3. Data

### Data Sources
**Impression logs:**
- User ID, ad ID, context (page, query)
- Whether ad was clicked (label)
- Timestamp, position, device

**User data:**
- Demographics (age, gender, location)
- Interests (inferred from behavior)
- Browsing history, past clicks
- Lookalike segments

**Ad data:**
- Creative (image, text, landing page)
- Advertiser, campaign, ad group
- Category (travel, finance, e-commerce)
- Historical performance (CTR, conversion rate)

**Context data:**
- Page content (for contextual targeting)
- Time of day, day of week
- Device type (mobile, desktop)

### Labeling
**Positive label:** User clicked the ad (clear signal)

**Negative label:** User saw ad but didn't click (but noisy)
- Position bias: Lower positions get fewer clicks regardless of quality
- Attention bias: User might not have seen the ad

**Attribution window:** How long to wait for click? (typically 24 hours)

### Feature Engineering

**User features:**
- Demographics (age_bin, gender, country)
- Historical CTR (user's past click rate on ads)
- Recency (time since last click)
- Engagement level (active vs passive user)

**Ad features:**
- Historical CTR (ad's past performance)
- Advertiser reputation
- Creative quality score
- Category embeddings

**Context features:**
- Page category (sports, news, shopping)
- Query keywords (for search ads)
- Time of day (ads perform differently morning vs night)

**Interaction features:**
- User-ad affinity (cosine similarity of embeddings)
- User-advertiser history (past clicks on this advertiser)
- Cross features (user_age × ad_category)

**Sequential features:**
- User's ad click sequence (LSTM/Transformer)
- Ad fatigue (seen this ad too many times?)

### Handling Class Imbalance
CTR is often <1%, so 99% of examples are negatives.

**Solutions:**
- Downsample negatives (keep all clicks, sample 10% of non-clicks)
- Class weights in loss function
- Focal loss (focuses on hard examples)

## 4. Model

### Model Evolution

**Stage 1: Logistic Regression**
- Linear model: `P(click) = sigmoid(w · x + b)`
- Pros: Fast, interpretable, well-calibrated
- Cons: Limited expressiveness, manual feature crosses

**Stage 2: Logistic Regression + Manual Feature Crosses**
- Add cross features: `user_age × ad_category`
- Still linear but more expressive
- Feature engineering heavy

**Stage 3: GBDT (Gradient Boosted Decision Trees)**
- XGBoost, LightGBM
- Automatically learns feature interactions
- Pros: Strong performance, handles missing values
- Cons: Calibration issues, doesn't use raw text/images

**Stage 4: Deep Learning**

**Factorization Machines (FM):**
- Models all pairwise feature interactions
- `y = w0 + Σ wi xi + Σ <vi, vj> xi xj`

**Wide & Deep:**
- Wide: memorize specific patterns (user ID × ad ID)
- Deep: generalize via embeddings
- Combines best of both

**Deep & Cross Network (DCN):**
- Explicitly models feature crosses at each layer
- Better than naive deep network for tabular data

**Two-tower model:**
- User tower: encode user features → user embedding
- Ad tower: encode ad features → ad embedding
- Score = dot product of embeddings
- Efficient for candidate generation (precompute ad embeddings)

### Architecture Choice
For large-scale ads systems:
- **Retrieval (candidate generation):** Two-tower model (fast, can use ANN search)
- **Ranking:** Wide & Deep or DCN (accurate CTR prediction)
- **Calibration layer:** Final sigmoid with temperature scaling

### Training Strategy

**Loss function:**
- Binary cross-entropy (log loss)
- Weighted by impression importance (higher weight for valuable users)

**Handling feedback delay:**
- Click may happen hours after impression
- Use importance sampling or delayed feedback modeling

**Negative downsampling:**
- Sample 10% of negatives
- Correct predictions: `p_corrected = p_model / (p_model + (1 - p_model) / r)`
  where `r` is downsampling rate

**Regularization:**
- L2 regularization (prevent overfitting to rare features)
- Dropout in deep networks

**Distributed training:**
- Data parallelism (split data across GPUs)
- Use parameter servers for large embedding tables

## 5. Serving

### Auction Flow
```
1. User lands on page
2. Ad request sent to auction server
3. Retrieve candidate ads (inverted index, targeting criteria)
4. Predict CTR for each candidate
5. Compute ad rank = bid × pCTR × quality_score
6. Run auction (charge second-price or first-price)
7. Return winning ad
```

### Latency Budget
- Total: <100ms (ads shouldn't slow down page load)
- Candidate retrieval: 20ms (target top 100-500 ads)
- CTR prediction: 50ms (batch inference for all candidates)
- Auction logic: 10ms
- Overhead: 20ms

### Real-Time Serving Challenges

**Challenge 1: Large embedding tables**
- User embeddings: 100M users × 128 dims = 50GB
- Ad embeddings: 10M ads × 128 dims = 5GB
- Solution: Distributed embedding lookup, caching

**Challenge 2: Feature freshness**
- User features change (just clicked an ad)
- Solution: Real-time feature computation (stream processing)

**Challenge 3: Model updates**
- Need to retrain frequently (data distribution shifts daily)
- Solution: Online learning or daily batch retraining

### Optimization Strategies

**Caching:**
- Cache ad embeddings (change infrequently)
- Cache popular user embeddings
- Cache predictions for common (user, ad) pairs

**Batching:**
- Score all candidate ads in one batch (GPU efficient)

**Model optimization:**
- Quantization (FP32 → INT8)
- Prune low-weight connections
- Distillation (compress large model into small one)

**Multi-stage ranking:**
- Stage 1: Fast two-tower model, narrow to top 100
- Stage 2: Accurate Wide & Deep model, final ranking

### A/B Testing
- **Randomization unit:** User (not impression, to avoid interference)
- **Duration:** 1-2 weeks (account for day-of-week effects)
- **Metrics:** CTR, revenue, user engagement
- **Interleaving:** Show control and treatment ads interleaved, see which gets more clicks

## 6. Monitoring

### Data Monitoring

**Feature drift:**
- User demographics shift (more mobile users)
- Ad creative trends change
- Detect with KL divergence on feature distributions

**Label drift:**
- CTR changes seasonally (holidays, events)
- New ad formats have different CTR

**Training-serving skew:**
- Features computed differently in training vs serving
- Example: "user_past_clicks" counted differently
- Solution: Use feature store with unified logic

### Model Monitoring

**Calibration drift:**
- Predicted CTR = 2%, observed CTR = 1% (model is overconfident)
- Causes: Data distribution shift, model staleness
- Solution: Recalibrate or retrain

**Performance by segment:**
- Model may underperform on mobile, or certain ad categories
- Track AUC/calibration per segment
- Retrain with segment-specific weights

**Cold start performance:**
- New ads have no historical CTR
- Fall back to ad category average or advertiser average

### Business Monitoring

**Revenue metrics:**
- RPM (revenue per 1000 impressions)
- eCPM (effective cost per mille)
- Advertiser spend

**User experience:**
- Ad load (don't show too many ads)
- User complaints (irrelevant ads, scams)

### Feedback Loops

**Positive feedback loop:**
- High pCTR ads get shown more, get more clicks, pCTR increases further
- Can suppress new ads (cold start problem)

**Negative feedback loop:**
- Bad ad gets shown, users don't click, pCTR drops, ad stops showing
- Good self-correction

**Mitigation:**
- Exploration: Randomly boost new ads (epsilon-greedy)
- Thompson sampling (Bayesian approach)

### Retraining Strategy

**Frequency:**
- Daily retraining (most common)
- Hourly for fast-moving campaigns (limited to small model updates)

**Online learning:**
- Update model in real-time as clicks come in
- Pros: Always fresh
- Cons: Can be unstable, harder to debug

**Shadow deployment:**
- Run new model alongside old model
- Compare predictions and business metrics
- Switch if new model is better

## Key Trade-offs

**AUC vs Calibration:**
- High AUC = good ranking
- Good calibration = accurate probabilities
- Ads need both (bid × pCTR requires calibrated probabilities)

**Personalization vs Privacy:**
- More user data = better predictions
- But raises privacy concerns (GDPR, CCPA)
- Solution: Differential privacy, on-device learning

**Exploration vs Exploitation:**
- Exploit: Show ads with high predicted CTR
- Explore: Try new ads to gather data
- Solution: Epsilon-greedy, Thompson sampling

**Latency vs Accuracy:**
- Heavy models are accurate but slow
- Solution: Multi-stage ranking (fast model → accurate model)

## Common Interview Follow-ups

**"How do you handle cold start?"**
- **New ads:** Use ad category CTR, advertiser historical CTR, content-based features
- **New users:** Use demographic averages, contextual features
- **Exploration:** Boost new ads temporarily to collect data

**"How do you prevent bad ads?"**
- **Policy filters:** Rule-based (block profanity, scams)
- **Quality score:** Penalize ads with low engagement or high bounce rate
- **Human review:** Sample high-spend ads for manual check

**"How do you handle ad fatigue?"**
- Track: How many times user saw this ad?
- Penalize: Reduce pCTR if user saw ad many times
- Frequency capping: Don't show same ad more than N times per day

**"How do you optimize for conversions, not just clicks?"**
- **Post-click modeling:** Predict P(conversion | click, user, ad)
- **Multi-task learning:** Joint model for CTR and CVR
- **Delayed feedback:** Conversion may happen days after click

**"How do you handle budget pacing?"**
- Advertiser sets daily budget
- Need to spread budget throughout day (not spend it all at midnight)
- Adjust bid multiplier based on spend rate

**"What if there's click fraud?"**
- **Detect:** Unusual click patterns (same IP, bot-like behavior)
- **Filter:** Remove suspicious clicks before training
- **Charge:** Don't charge advertisers for fraudulent clicks
