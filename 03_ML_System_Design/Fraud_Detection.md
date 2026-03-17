# Fraud Detection System

Design a system to detect fraudulent transactions in real-time (e.g., credit card fraud, payment fraud, fake accounts).

## 1. Problem Definition

### Clarifying Questions
- Fraud type? (payment fraud, account takeover, identity theft, promotion abuse)
- Scale? (transactions per second, fraud rate)
- Latency requirements? (real-time blocking, post-transaction review)
- Costs? (false positive = angry customer, false negative = lost money)
- Existing systems? (rule-based filters, manual review team)

### Scope
- Focus: Real-time fraud scoring for transactions
- In scope: Model training, feature engineering, serving, human review
- Out of scope: User authentication, payment processing infrastructure

### Business Context
This is a cost-sensitive problem:
- **False positive (block legitimate transaction):** Customer frustration, lost revenue, support cost
- **False negative (miss fraud):** Direct financial loss, chargeback fees, reputation damage

The optimal trade-off depends on transaction value and user history.

## 2. Metrics

### Offline Metrics
- **Precision:** Of flagged transactions, how many are truly fraudulent?
- **Recall:** Of all fraudulent transactions, how many did we catch?
- **F1 score / F-beta:** Balance precision and recall (or weight recall higher)
- **AUC-PR:** Better than AUC-ROC for imbalanced data
- **Cost-weighted metric:** `cost = FP_cost × FP_count + FN_cost × FN_count`

### Online Metrics
- **Fraud caught rate:** % of fraud detected before payout
- **False positive rate:** % of legitimate transactions blocked
- **Monetary loss prevented:** Dollar amount of fraud caught
- **Chargeback rate:** % of transactions that result in chargeback
- **Customer friction:** % of users who fail transaction and don't retry

### Guardrail Metrics
- **Latency:** Fraud check < 100ms (don't slow down checkout)
- **Model coverage:** % of transactions scored by model (vs fallback rules)
- **Human review backlog:** Queue depth for manual review
- **Demographic fairness:** Avoid bias against certain groups (geography, demographics)

### Key Principle
Optimize for monetary loss prevented, not just accuracy. A $10K fraud is worse than a $10 fraud. Consider transaction value in metrics.

## 3. Data

### Data Sources

**Transaction data:**
- Amount, currency, merchant, timestamp
- Payment method (card, bank transfer, crypto)
- Device info (IP, user agent, device ID)
- Location (billing address, shipping address, IP geolocation)

**User behavior:**
- Account age, activity history
- Past transactions (frequency, amounts)
- Login patterns (time of day, devices)
- Profile completeness (verified email, phone)

**Fraud labels:**
- Explicit fraud (chargebacks, user reports, law enforcement)
- Implicit signals (account closed, payment reversed)
- Delayed feedback (chargeback arrives weeks later)

**External signals:**
- Device fingerprinting (shared devices across accounts)
- IP reputation (known VPN, proxy, datacenter)
- Email/phone reputation (disposable email, VoIP)
- Blocklists (stolen cards, known fraudsters)

### Handling Extreme Class Imbalance
Fraud rate is typically 0.1-1%, so 99%+ of transactions are legitimate.

**Problems:**
- Model defaults to "predict everything as legit"
- Offline metrics (accuracy) are misleading
- Hard to learn rare fraud patterns

**Solutions:**
- Oversample fraud cases (duplicate or SMOTE)
- Undersample legit cases (but keep hard negatives)
- Class weights in loss function (weight fraud 100x higher)
- Focal loss (focus on hard examples)
- Anomaly detection (model "normal" behavior, flag deviations)

### Feature Engineering

**Transaction features:**
- Amount (absolute, relative to user's typical spending)
- Time of day, day of week
- Merchant category (high-risk: gift cards, crypto)
- International transaction (cross-border is higher risk)

**Velocity features (key for fraud detection):**
- Transactions in last 1 hour, 24 hours, 7 days
- Amount spent in last 1 hour, 24 hours
- Number of failed transactions (fraudsters try multiple cards)
- Number of unique merchants, locations

**User history features:**
- Account age (new accounts are riskier)
- Past fraud rate (user's historical fraud %)
- Time since last transaction
- Average transaction amount

**Device/location features:**
- New device (never seen before)
- New location (IP geolocation)
- Device fingerprint (browser, OS, screen resolution)
- Impossible travel (transaction in NYC, then London 1 hour later)

**Behavioral anomaly features:**
- Deviation from user's typical behavior (amount, merchant type)
- Deviation from global patterns
- Time since account creation (new accounts doing large transactions)

**Network features (graph-based):**
- Shared devices across accounts (collusion)
- Shared payment methods
- Shared IP addresses
- Connected to known fraudsters

### Handling Delayed Labels
- Transaction happens at time T
- Chargeback happens at time T+30 days
- Model trained on T sees label only at T+30

**Solutions:**
- Train on historical data (labels are complete)
- Importance sampling (weight recent data lower, older data higher)
- Two-stage modeling (immediate fraud indicators + delayed chargeback prediction)

## 4. Model

### Multi-Stage Architecture

**Stage 1: Rule-based Filters (Pre-filters)**
Fast, deterministic rules catch obvious fraud:
- Transaction amount > $10K (high risk)
- Known stolen card (blocklist)
- IP from blacklisted datacenter
- Velocity exceeded (e.g., 10 transactions in 1 minute)

**Stage 2: ML Scoring**
Score each transaction with fraud probability.

**Model choice:**
- **Logistic Regression:** Fast, interpretable, well-calibrated baseline
- **GBDT (XGBoost, LightGBM):** Strong performance, handles features well
- **Neural Network:** For complex patterns, but overkill for tabular data
- **Anomaly Detection:** Isolation Forest, Autoencoder (model normal, flag outliers)

**Training:**
- Loss: Binary cross-entropy with class weights
- Or: Cost-sensitive loss (weight by transaction amount)
- Regularization: L2 or early stopping (fraud patterns change fast)

**Ensemble:**
- Combine rule-based score + ML score
- Reduces variance, handles both known and novel fraud

**Stage 3: Human Review**
- Transactions with medium fraud score (0.3-0.7) go to manual review
- High-value transactions always reviewed (even if low score)
- Specialized teams for different fraud types

### Model Architectures

**Supervised Learning (most common):**
- Train on labeled transactions (fraud vs legit)
- Features: transaction, user, device, velocity
- Output: Fraud probability

**Anomaly Detection:**
- Train on legitimate transactions only (unsupervised)
- Flag transactions that deviate from normal
- Good for novel fraud types (zero-day attacks)

**Graph Neural Network:**
- Model relationships (user-device, user-IP, user-merchant)
- Detect fraud rings (coordinated attack by multiple accounts)
- More complex, used by advanced systems (PayPal, Stripe)

**Time-series Model:**
- Model user behavior over time (RNN, LSTM)
- Detect sudden behavioral change
- Expensive, mostly for high-value use cases

### Handling Concept Drift
Fraudsters adapt quickly. Patterns that worked last month may not work this month.

**Solutions:**
- Retrain frequently (daily or weekly)
- Online learning (update model in real-time)
- Ensemble of models trained on different time windows
- Monitor performance by cohort (recent vs old data)

## 5. Serving

### Real-Time Scoring
```
Transaction initiated
  → Extract features (user history, device, velocity)
  → Pre-filters (rule-based checks)
  → ML model scoring
  → Compute final risk score
  → Decision:
      - Low risk (score < 0.1): Approve
      - Medium risk (0.1-0.5): Step-up auth (2FA, CAPTCHA)
      - High risk (> 0.5): Block or manual review
```

### Latency Budget
- Total: <100ms (don't slow down checkout)
- Feature extraction: 30ms (query databases for user history)
- Model scoring: 30ms (GBDT inference)
- Decision logic: 10ms
- Logging: 10ms
- Overhead: 20ms

### Optimization Strategies

**Caching:**
- Cache user features (update every 5 minutes, not real-time)
- Cache device reputation (update daily)
- Pre-compute velocity features (stream processing)

**Feature store:**
- Centralized store for real-time features
- Ensures train-serve consistency
- Examples: Feast, Tecton

**Model optimization:**
- GBDT is already fast (<10ms for inference)
- Quantization if using neural networks
- Pruning (remove low-importance trees)

**Asynchronous processing:**
- Approve transaction immediately (for low-risk)
- Run expensive checks asynchronously (graph analysis, external API calls)
- Reverse transaction if later flagged

**Multi-stage decision:**
- Stage 1: Fast model (logistic regression) filters 95% as safe
- Stage 2: Expensive model (GNN) scores risky 5%
- Saves compute

### Handling False Positives
False positives hurt legitimate users.

**Step-up authentication:**
- Don't block outright, ask for 2FA
- Or ask security questions
- Reduces friction vs hard block

**Adaptive thresholds:**
- Lower threshold for high-value transactions
- Higher threshold for trusted users
- Time-based (lower threshold at 3am, higher during day)

**User feedback:**
- "Was this transaction fraudulent?" (ask user)
- Feed back into training data
- Reduces false positives over time

### A/B Testing
- Control: Existing fraud system
- Treatment: New ML model
- Metrics: Fraud caught rate, false positive rate, monetary loss
- Be careful: Can't fully randomize (fraud is rare, need large sample)
- Use stratified testing (high-risk segment only)

## 6. Monitoring

### Data Monitoring

**Feature drift:**
- Transaction amounts increase (inflation, user behavior change)
- Device distribution changes (more mobile, less desktop)
- Detect with KL divergence on feature distributions

**Label drift:**
- Fraud patterns evolve (new attack vectors)
- Chargeback labels arrive late (monitor label lag)

**Velocity feature staleness:**
- If stream processing lags, velocity features are stale
- Monitor data freshness (time lag between transaction and feature computation)

### Model Monitoring

**Performance by segment:**
- High-value vs low-value transactions
- New users vs old users
- Geography (fraud patterns differ by country)
- Track precision/recall per segment

**Concept drift detection:**
- Model trained on Jan data may underperform in Feb
- Track AUC over time, alert if drops
- Retrain when performance degrades

**Cold start performance:**
- New users have no history (model relies on device, location)
- Track performance on new vs returning users

### Online Metrics

**Fraud caught rate:**
- % of fraud detected by model (vs missed)
- Goal: >90% (but depends on business)

**False positive rate:**
- % of legitimate transactions flagged
- Goal: <1% (but depends on risk tolerance)

**Chargeback rate:**
- % of transactions that result in chargeback
- Lagging indicator (chargebacks arrive weeks later)

**Monetary loss:**
- Dollar amount of fraud missed
- More important than raw fraud count

### Business Monitoring

**Customer friction:**
- % of users who fail transaction and don't retry
- Track step-up auth success rate (do users complete 2FA?)

**Human review metrics:**
- Queue depth (backlog of transactions awaiting review)
- Reviewer decision distribution (are we sending too much to review?)
- Reviewer agreement with model (is model flagging the right stuff?)

**Cost metrics:**
- Cost of fraud (lost money + chargeback fees)
- Cost of false positives (lost sales + support)
- Cost of operation (human review, compute)

### Feedback Loops

**Risk:**
- Fraud caught by model gets labeled as fraud
- Fraud missed by model never gets labeled
- Model trains on biased labels (selection bias)

**Mitigation:**
- Random sampling for manual review (not just high-score transactions)
- External data (law enforcement, chargeback data)
- Regular audits of missed fraud

**Adversarial feedback loop:**
- Fraudsters learn model's weaknesses
- Adapt attack strategies
- Model degrades over time

**Mitigation:**
- Retrain frequently
- Ensemble diverse models (harder to evade all)
- Keep some detection logic secret (don't reveal exact thresholds)

### Retraining Strategy
- **Frequency:** Weekly (fraud patterns change fast)
- **Online learning:** Real-time updates (risky, can be unstable)
- **Incremental learning:** Update model with recent data only
- **Shadow mode:** Run new model alongside old, compare before switching

## Key Trade-offs

**Precision vs Recall:**
- High recall = catch all fraud, but block many legitimate transactions
- High precision = minimize false positives, but miss some fraud
- **Recommendation:** Tune by transaction value. High-value = favor recall. Low-value = favor precision.

**Latency vs Accuracy:**
- Complex models (GNN, LSTM) are accurate but slow
- Simple models (logistic regression) are fast but less accurate
- **Solution:** Multi-stage (fast model → slow model for risky transactions)

**Automation vs Human Review:**
- Full automation = scalable but error-prone
- Human review = accurate but slow and expensive
- **Solution:** Automate obvious cases, humans review uncertain ones

**Immediate Block vs Step-up Auth:**
- Blocking = safe but frustrating
- Step-up auth (2FA) = better UX but slower and some users won't complete
- **Solution:** Block high-risk, step-up for medium-risk, approve low-risk

## Common Interview Follow-ups

**"How do you handle cold start (new users)?"**
- No historical features (account age = 0, past transactions = 0)
- Rely on device features (IP reputation, device fingerprint)
- Start with conservative threshold (lower risk tolerance)
- Collect data, relax threshold as user builds history

**"How do you handle adversarial attacks?"**
- Fraudsters test the model (try small transactions to find threshold)
- Adaptive attacks (change behavior to evade detection)
- **Mitigation:** Randomize threshold slightly, ensemble models, keep detection logic opaque

**"How do you balance fraud prevention and user experience?"**
- False positives hurt UX (customers can't complete purchase)
- Use step-up auth instead of blocking
- Personalize threshold (trusted users = higher threshold)
- Explain decisions transparently ("We need to verify your identity")

**"What if fraud patterns change suddenly?"**
- New attack vector emerges (e.g., synthetic identity fraud)
- Model may not catch it initially
- **Mitigation:** Anomaly detection (catch novel patterns), rapid retraining, human review for high-risk

**"How do you handle fraud rings (coordinated attacks)?"**
- Multiple accounts, shared devices/IPs, coordinated timing
- Graph Neural Network (model connections)
- Network features (detect shared resources)
- Clustering (find groups of suspicious accounts)

**"How do you measure ROI of fraud detection system?"**
- Benefit: Fraud prevented ($ saved)
- Cost: False positives (lost sales), human review, infrastructure
- ROI = (fraud prevented - costs) / costs
- Also consider brand reputation (hard to quantify)

**"What if chargebacks arrive late?"**
- Labels are delayed by weeks
- Can't train on most recent data
- **Solutions:** Train on older data (complete labels), use immediate signals (account closed, payment failed), two-stage modeling
