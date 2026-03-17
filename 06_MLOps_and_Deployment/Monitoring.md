# Monitoring ML Systems in Production

## Overview

ML models degrade over time. The world changes, but the model doesn't (unless you retrain). Monitoring catches issues early before they impact users or the business. This document covers what to monitor, how to detect problems, and when to retrain.

## 1. What Can Go Wrong?

### Data Drift
Feature distributions change between training and serving.

**Examples:**
- E-commerce: User demographics shift (more mobile users)
- Search: Query distribution changes (new trending topics)
- Fraud: Attack patterns evolve (fraudsters adapt)

**Why it's bad:**
- Model was trained on old distribution
- May not generalize to new distribution
- Predictions become less reliable

### Concept Drift
The relationship between features and target changes.

**Examples:**
- Recommendation: User preferences change (pandemic shifts viewing habits)
- Housing prices: What predicts price changes (interest rates change)
- Spam detection: What makes an email spam evolves (new tactics)

**Why it's bad:**
- Even if features stay the same, their predictive power changes
- Model is literally wrong about the world

### Label Drift
Ground truth distribution changes.

**Examples:**
- Fraud: Fraud rate increases (new attack wave)
- Content moderation: More toxic content (election season)

**Different from concept drift:**
- Features → label relationship may stay same
- But base rate changes (affects precision/recall trade-off)

### Train-Serve Skew
Features computed differently in training vs serving.

**Examples:**
- "Past 7 days clicks" computed on different time windows
- Missing features (feature store down, fallback to default)
- Preprocessing differs (normalization parameters)

**Why it's bad:**
- Model sees different input than what it was trained on
- Can cause sudden performance drop

### Model Staleness
Model becomes outdated as time passes.

**Examples:**
- Recommendation: New products aren't in training data (cold start)
- NLP: New slang, memes, topics emerge
- Ad CTR: Seasonal patterns (holiday shopping)

**Why it's bad:**
- Model can't handle new entities or patterns
- Performance gradually degrades

## 2. Monitoring Strategies

### Data Monitoring

**Feature Distribution Monitoring**

Compare serving features to training features.

**Metrics:**
- **KL Divergence:** `D(P||Q) = Σ P(x) log(P(x)/Q(x))`
  - Measures difference between distributions
  - Higher = more drift
  - Problem: Undefined if Q(x) = 0 for some x

- **Population Stability Index (PSI):**
  - Similar to KL divergence but symmetric
  - `PSI = Σ (P(x) - Q(x)) × log(P(x)/Q(x))`
  - Industry standard: PSI < 0.1 (no drift), 0.1-0.25 (moderate), > 0.25 (significant)

- **Chi-squared test (categorical features):**
  - Compare observed vs expected counts
  - Returns p-value (p < 0.05 = significant drift)

- **Kolmogorov-Smirnov test (continuous features):**
  - Compare cumulative distributions
  - Returns p-value

**Implementation:**
```python
# Example: Compute PSI
def compute_psi(expected, actual, bins=10):
    expected_hist, _ = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_hist / len(expected)
    actual_pct = actual_hist / len(actual)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi
```

**What to track:**
- Per-feature drift (which features are drifting?)
- Overall drift (aggregate across features)
- Drift over time (is it getting worse?)

**Alerting thresholds:**
- PSI > 0.25: Alert (significant drift)
- Multiple features drifting: Alert (systemic issue)

**Missing Data Monitoring**

Track null rates per feature.

**Metrics:**
- Null rate: % of samples with missing value
- Compare to training null rate

**Alerts:**
- Null rate > 2x training null rate
- New features showing up null (integration bug)

**Schema Validation**

Check that data matches expected schema.

**Checks:**
- Data type (string vs int)
- Range (age shouldn't be negative)
- Cardinality (category should have <100 values)
- Required fields (user_id must be present)

**Tools:**
- TensorFlow Data Validation (TFDV)
- Great Expectations
- Custom checks

### Model Monitoring

**Prediction Distribution Monitoring**

Monitor what the model is predicting.

**Metrics:**
- Mean prediction (is model predicting higher/lower than before?)
- Prediction variance (is model more/less confident?)
- Distribution by bin (how many predictions in [0-0.1], [0.1-0.2], etc.)

**Why it matters:**
- Sudden shift = something changed (data or model)
- If model predicts 90% of samples as class 0 (but training was 50/50), something's wrong

**Example alert:**
- Mean prediction shifted by >10%
- >20% of predictions are in extreme bins (0-0.01 or 0.99-1.0)

**Calibration Monitoring**

Check if predicted probabilities match observed frequencies.

**How to measure:**
- Bin predictions (0-0.1, 0.1-0.2, ..., 0.9-1.0)
- For each bin, compute observed frequency
- Well-calibrated: Predicted 30% = Observed 30%

**Metrics:**
- Expected Calibration Error (ECE): Average absolute difference
- Brier score: `(predicted - actual)²`

**Why it matters:**
- Calibration affects decision-making (e.g., bid × pCTR in ads)
- Drift can break calibration even if ranking is still good

**Performance Monitoring**

Track actual model performance on live data (requires labels).

**Challenge:**
- Labels arrive late (e.g., fraud chargebacks come weeks later)
- Not all samples have labels (not all users give feedback)

**Approaches:**
- Sample and label (manual or automated)
- Use proxy labels (clicks for search, even though not perfect)
- Use delayed labels (fraud labels arrive in 30 days, track lagging metric)

**Metrics:**
- Same as offline evaluation (AUC, precision, recall, F1)
- Track over time (is performance degrading?)

**Alerting:**
- AUC drops by >5%
- Precision or recall drops below threshold

### Infrastructure Monitoring

**Latency**

Track end-to-end latency (how long to return prediction).

**Metrics:**
- p50, p95, p99 latency (percentiles matter more than average)
- By component (data fetch, model inference, post-processing)

**Alerts:**
- p99 > SLA (e.g., >200ms for search, >100ms for fraud)
- Latency spike (2x increase)

**Throughput**

Track queries per second (QPS).

**Metrics:**
- Current QPS
- QPS capacity (max throughput before degradation)

**Alerts:**
- QPS approaching capacity (need to scale)
- QPS drop (are requests failing?)

**Error Rate**

Track failed requests.

**Types:**
- Model errors (exception during inference)
- Data errors (missing features, schema mismatch)
- Timeout errors (model too slow)

**Alerts:**
- Error rate > 1%
- Any 500 errors (critical)

**Resource Utilization**

Track CPU, memory, GPU usage.

**Metrics:**
- CPU %
- Memory %
- GPU utilization

**Alerts:**
- CPU > 80% sustained (need to scale)
- Memory leak (usage increasing over time)

### Business Monitoring

**Online Metrics**

Track actual business impact (requires A/B test or tracking).

**Examples:**
- Recommendation: Click rate, watch time
- Ads: Revenue, CTR
- Search: Success rate, abandonment rate
- Fraud: Fraud caught, false positive rate

**Why it matters:**
- Offline metrics don't always predict online metrics
- Model can have good AUC but hurt revenue

**Alerting:**
- Online metric drops by >5%
- Guardrail metric violated (e.g., latency too high)

**Cost Monitoring**

Track cost per prediction (especially for LLMs).

**Metrics:**
- Inference cost (compute)
- Token cost (for LLM APIs)
- Data storage cost
- Human review cost

**Why it matters:**
- Cost can balloon unexpectedly (e.g., model gets called 10x more)

## 3. Alerting Strategies

### Threshold-Based Alerts

Set fixed thresholds for metrics.

**Example:**
- If AUC < 0.85: Alert
- If latency p99 > 200ms: Alert
- If error rate > 1%: Alert

**Pros:**
- Simple, easy to understand
- Actionable (clear threshold)

**Cons:**
- Hard to set good thresholds (too sensitive = alert fatigue, too loose = miss issues)
- Doesn't adapt to trends

### Anomaly Detection Alerts

Learn normal behavior, alert on deviations.

**Methods:**
- Z-score (alert if |value - mean| > 3 × std)
- Moving average (alert if value deviates from 7-day moving avg)
- Seasonal decomposition (account for day-of-week, hourly patterns)
- ML-based (isolation forest, LSTM autoencoder)

**Pros:**
- Adapts to trends and seasonality
- Catches unexpected issues

**Cons:**
- Can be noisy (false alarms)
- Less interpretable

### Rate-of-Change Alerts

Alert on sudden changes.

**Example:**
- AUC dropped by >5% in 1 hour
- QPS doubled in 10 minutes (unexpected traffic spike)

**Pros:**
- Catches sudden issues (deployments, data pipeline bugs)

**Cons:**
- Misses gradual degradation

### Composite Alerts

Combine multiple signals.

**Example:**
- Alert if (AUC drops by >3%) AND (feature drift PSI > 0.25)
- Reduces false positives (both must be true)

### Alert Prioritization

Not all alerts are equal.

**Severity levels:**
- **P0 (Critical):** Model is down, serving errors
- **P1 (High):** Performance degraded significantly, affecting users
- **P2 (Medium):** Drift detected, but model still functional
- **P3 (Low):** Minor anomaly, investigate when convenient

**On-call rotation:**
- P0/P1: Page immediately
- P2: Slack notification
- P3: Email digest

## 4. Feedback Loops

### Positive Feedback Loop

Model predictions influence future data, which reinforces the predictions.

**Example:**
- Recommendation: Model recommends popular videos
- Users watch popular videos
- Videos become more popular
- Model recommends them even more (filter bubble)

**Why it's bad:**
- Reduces diversity
- Suppresses new content
- Hurts long-term engagement

**Mitigation:**
- Exploration (randomly recommend less-popular items)
- Diversity penalties in ranking
- Periodically retrain on randomized serving data

### Negative Feedback Loop

Model predictions cause behavior that invalidates the model.

**Example:**
- Traffic prediction: Model predicts traffic jam on route A
- Navigation app routes users to route B
- Route A is now clear (prediction was wrong)

**Why it's bad:**
- Model's predictions change the world in a way that makes them wrong

**Mitigation:**
- Model the feedback (predict user behavior given predictions)
- Use bandit algorithms (exploration + exploitation)

### Adversarial Feedback Loop

Adversaries learn model's behavior and exploit it.

**Example:**
- Fraud detection: Fraudsters test small transactions to find threshold
- Once they know threshold, they stay just below it

**Why it's bad:**
- Model becomes predictable
- Adversaries optimize against it

**Mitigation:**
- Randomize thresholds slightly
- Keep detection logic secret (don't reveal exact features/thresholds)
- Ensemble models (harder to evade multiple models)
- Continuous retraining (adapt to new tactics)

## 5. Retraining Strategies

### When to Retrain?

**Scheduled Retraining**

Retrain on a fixed schedule.

**Common schedules:**
- Daily (fraud, ads, fast-moving content)
- Weekly (search, recommendations)
- Monthly (slower-moving models)

**Pros:**
- Predictable, easy to automate
- Ensures model stays fresh

**Cons:**
- May retrain unnecessarily (if no drift)
- May not retrain fast enough (if sudden drift)

**Drift-Triggered Retraining**

Retrain when drift is detected.

**Triggers:**
- Feature drift (PSI > threshold)
- Performance drop (AUC < threshold)
- Prediction shift (mean prediction changed)

**Pros:**
- Retrain only when needed (saves compute)
- Responsive to sudden changes

**Cons:**
- Need good drift detection
- May be too reactive (wait until performance drops)

**Performance-Triggered Retraining**

Retrain when online metrics degrade.

**Triggers:**
- CTR drops by >5%
- Revenue drops by >X%
- User complaints increase

**Pros:**
- Directly tied to business impact

**Cons:**
- Delayed (performance already hurt before retraining)
- Hard to attribute (is it model or something else?)

**Hybrid Approach (Recommended)**

Combine strategies:
- Scheduled retraining (baseline, e.g., weekly)
- Drift monitoring (early warning)
- Performance monitoring (trigger emergency retrain)

### What to Retrain?

**Full Retraining**

Retrain model from scratch on new data.

**Pros:**
- Learns latest patterns
- Clean slate (no accumulated errors)

**Cons:**
- Expensive (requires full training pipeline)
- Slow (can take hours/days for large models)

**Incremental Retraining**

Update model with new data (online learning, fine-tuning).

**Pros:**
- Fast (minutes instead of hours)
- Continuously adapts

**Cons:**
- Can accumulate errors (catastrophic forgetting)
- Harder to debug
- May not fully adapt to major shifts

**When to use:**
- Full retraining: Major changes (new features, large drift)
- Incremental: Minor updates (daily patterns, small drift)

### Retraining Pipeline

1. **Data collection:** Gather recent data (logs, labels)
2. **Feature engineering:** Compute features (use same pipeline as training)
3. **Training:** Train new model on recent data
4. **Evaluation:** Test on holdout set (offline metrics)
5. **Validation:** Compare to old model (online metrics via shadow deployment)
6. **Deployment:** If new model is better, roll out (canary → full)
7. **Monitoring:** Track performance of new model

**Automation:**
- Use orchestration tools (Airflow, Kubeflow, Prefect)
- Trigger retraining automatically (on schedule or drift)
- Auto-deploy if validation passes (or require human approval for safety)

## 6. Shadow Deployment

### What Is It?

Run new model alongside old model. Users see old model's predictions, but log both models' predictions.

### Why?

- Compare models in production (apples-to-apples)
- No user impact (safe to test)
- Catch bugs before rollout

### Process

1. Deploy new model in shadow mode
2. For each request, score with both models
3. Serve old model's prediction (user sees this)
4. Log both predictions + features + labels
5. After 1-2 weeks, analyze:
   - Offline metrics (which model is more accurate?)
   - Latency (is new model slower?)
   - Prediction agreement (how often do they agree?)
6. If new model is better, roll out (canary → full)

### What to Compare

**Offline metrics:**
- AUC, precision, recall (on labeled subset)

**Prediction distribution:**
- Are predictions similar? (if too different, investigate)

**Latency:**
- Is new model slower? (acceptable trade-off?)

**Edge cases:**
- How do models handle rare inputs?
- Are there systematic differences?

**Agreement rate:**
- % of predictions where models agree (e.g., same class)
- Low agreement = models are very different (risky)

## 7. Monitoring Tools

### Open-Source

**Evidently AI:**
- Data drift detection
- Model performance monitoring
- Interactive dashboards

**Fiddler:**
- Model explainability + monitoring
- Drift detection, performance tracking

**Alibi Detect:**
- Drift detection algorithms (KS, MMD, etc.)

**Great Expectations:**
- Data validation and profiling

### Cloud Platforms

**AWS SageMaker Model Monitor:**
- Drift detection
- Baseline comparison
- CloudWatch integration

**Google Vertex AI Model Monitoring:**
- Skew and drift detection
- Feature attribution monitoring

**Azure ML Model Monitoring:**
- Data drift tracking
- Model performance

### Custom Dashboards

Most companies build custom dashboards.

**Stack:**
- Logging: Prometheus (metrics), ELK (logs)
- Dashboards: Grafana, Kibana
- Alerting: PagerDuty, Slack

**What to visualize:**
- Time series (metrics over time)
- Distributions (feature histograms)
- Correlations (feature vs performance)
- Alerts (recent alerts and their status)

## 8. Case Study: Search Ranking Monitoring

**Data Monitoring:**
- Query distribution (are users searching for new topics?)
- Click distribution (are certain positions getting more clicks?)
- Feature drift (is user behavior changing?)

**Model Monitoring:**
- NDCG on human-labeled test set (monthly)
- Prediction distribution (are scores well-calibrated?)
- Calibration (do predicted CTR match observed CTR?)

**Business Monitoring:**
- CTR (click-through rate)
- Success rate (% of queries with satisfying click)
- Zero-result rate (% of queries with no results)

**Infrastructure Monitoring:**
- Latency (p99 < 200ms)
- QPS (queries per second)
- Error rate

**Retraining:**
- Weekly scheduled retraining (on latest query logs)
- Drift-triggered retraining (if query distribution shifts significantly)
- Shadow deployment (1 week) before rollout

**Alerts:**
- P0: Error rate > 5%, p99 latency > 500ms
- P1: CTR drops by >10%, zero-result rate > 5%
- P2: NDCG drops by >5%, feature drift PSI > 0.25

## Best Practices

1. **Start simple:** Monitor key metrics first (accuracy, latency, errors), add more later
2. **Baseline everything:** Need training distribution to compare serving distribution
3. **Automate alerts:** Don't rely on manual checks
4. **Prioritize alerts:** Not everything is urgent
5. **Document incidents:** When alerts fire, record what happened and how you fixed it
6. **Review regularly:** Weekly review of monitoring dashboards (catch slow degradation)
7. **Test your monitoring:** Inject synthetic drift, see if alerts fire
8. **Keep humans in loop:** Automation is good, but humans need to understand what's happening
