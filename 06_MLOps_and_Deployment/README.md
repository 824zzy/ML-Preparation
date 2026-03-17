# MLOps and Deployment

## Overview

This section covers production ML concerns: model serving, A/B testing, monitoring, and debugging production issues. These questions separate ML engineers who have shipped models from those who've only done Kaggle.

## Decision Table for Common Questions

| Question | Key Considerations | Files to Review |
|----------|-------------------|-----------------|
| "How do you deploy a model?" | Batch vs real-time, latency, scale | `Model_Serving.md` |
| "How do you A/B test a model?" | Randomization, metrics, duration | `AB_Testing.md` |
| "How do you monitor models?" | Data drift, performance, alerts | `Monitoring.md` |
| "What causes train-serve skew?" | Feature computation differences | `Model_Serving.md`, `Monitoring.md` |
| "How do you handle model staleness?" | Retraining frequency, online learning | `Monitoring.md` |
| "How do you reduce latency?" | Caching, quantization, batching | `Model_Serving.md` |

## Topics Covered

### Model Serving (`Model_Serving.md`)
- Batch vs real-time inference
- Serving architectures (model server, embedded, edge)
- Latency optimization (quantization, pruning, distillation)
- Caching strategies
- Load balancing and scaling
- Train-serve skew prevention

### A/B Testing (`AB_Testing.md`)
- Experimental design (randomization unit, sample size)
- Statistical significance (t-test, p-value, confidence intervals)
- Common pitfalls (peeking, novelty effect, Simpson's paradox)
- Multi-armed bandits (exploration vs exploitation)
- Interleaving experiments
- When NOT to A/B test

### Monitoring (`Monitoring.md`)
- Data drift detection (feature distribution shift)
- Concept drift (label distribution shift)
- Model performance degradation
- Alerting strategies (threshold-based, anomaly detection)
- Retraining triggers (scheduled, drift-based, performance-based)
- Shadow deployment (compare new vs old model)
- Feedback loops (positive, negative, adversarial)

## MLOps Stack Overview

**Experiment Tracking:**
- MLflow, Weights & Biases, Neptune
- Track metrics, hyperparameters, artifacts

**Feature Store:**
- Feast, Tecton, Hopsworks
- Ensure train-serve consistency
- Share features across teams

**Model Registry:**
- MLflow, SageMaker Model Registry
- Version models, track lineage

**Serving:**
- TensorFlow Serving, TorchServe, Triton
- API gateways (FastAPI, BentoML)
- Cloud services (SageMaker, Vertex AI)

**Monitoring:**
- Evidently AI, Fiddler, Arize
- Prometheus + Grafana (infrastructure)
- Custom dashboards

**Orchestration:**
- Airflow, Kubeflow, Prefect
- Schedule retraining, data pipelines

## Key Concepts

### Train-Serve Skew
The model behaves differently in production than in training.

**Causes:**
- Feature computed differently (e.g., "past_7_days_clicks" counted differently)
- Missing features at serving time
- Data preprocessing differs
- Model version mismatch

**Prevention:**
- Use feature store (single source of truth)
- Integration tests (compare train vs serve outputs)
- Log serving inputs, replay them in training

### Model Staleness
The model degrades over time because the world changes.

**Examples:**
- New products (e-commerce recommendation)
- Seasonal shifts (holiday shopping patterns)
- Adversarial adaptation (fraudsters learn model)

**Solutions:**
- Scheduled retraining (daily, weekly)
- Drift-triggered retraining (retrain when drift detected)
- Online learning (update in real-time, but risky)

### Shadow Deployment
Run new model alongside old model, compare predictions before switching.

**Benefits:**
- No user impact (users still see old model)
- Collect metrics on new model
- Debug issues safely

**Process:**
1. Deploy new model in shadow mode
2. Log predictions from both models
3. Compare offline metrics (accuracy, latency)
4. If new model is better, gradually ramp up traffic
5. Monitor for issues, rollback if needed

### Canary Deployment
Gradually roll out new model to small % of traffic.

**Process:**
1. Route 5% of traffic to new model
2. Monitor metrics (errors, latency, business metrics)
3. If good, increase to 25%, then 50%, then 100%
4. If bad, rollback immediately

## Common Interview Questions

**"How would you deploy a recommendation model?"**
- Batch precompute recommendations (daily), store in cache
- Real-time re-ranking based on recent activity
- Hybrid approach (batch candidates + real-time ranking)

**"How would you deploy a fraud detection model?"**
- Real-time scoring (latency < 100ms)
- Feature caching (pre-compute user features)
- Fallback to rules if model times out

**"How would you A/B test a search ranking model?"**
- Randomize by user (not query)
- Interleaving (show results from both models interleaved)
- Track CTR, time to first click, success rate
- Run for 1-2 weeks (account for day-of-week effects)

**"How do you know when to retrain?"**
- Scheduled (daily, weekly) - simple, predictable
- Drift-triggered (when feature distributions shift)
- Performance-triggered (when online metrics degrade)
- Combination (schedule + drift checks)

**"What metrics do you monitor in production?"**
- Data metrics (feature distributions, null rates)
- Model metrics (prediction distribution, confidence)
- Performance metrics (accuracy, precision, recall on live data)
- Infrastructure metrics (latency, QPS, error rate)
- Business metrics (revenue, engagement, conversions)

**"How do you handle model versioning?"**
- Model registry (MLflow, SageMaker)
- Semantic versioning (v1.2.3)
- Track training data, code, hyperparameters
- A/B test before replacing old version

**"What causes high latency in model serving?"**
- Large model size (use quantization, pruning, distillation)
- Complex preprocessing (cache features)
- Cold start (keep models warm, use predictive scaling)
- External API calls (batch them, use async)
- No batching (batch requests for GPU efficiency)

**"How do you debug a model that works in training but not production?"**
- Train-serve skew (check feature computation)
- Data drift (compare train vs serve distributions)
- Label leakage (using future information in training)
- Sampling bias (train on different population than serve)
- Version mismatch (wrong model/code/data)

## Anti-Patterns to Avoid

**Only monitoring accuracy:**
- Need to monitor latency, cost, fairness, etc.

**Not tracking lineage:**
- "Which training data produced this model?"
- "What hyperparameters were used?"

**No rollback plan:**
- Always have a way to quickly revert to old model

**Ignoring edge cases:**
- What if feature store is down?
- What if model times out?
- Have fallback logic (rules, cached predictions)

**Over-optimizing latency:**
- Sometimes 200ms is fine, don't spend weeks to get to 100ms
- Focus on user-facing metrics

**Retraining too frequently:**
- Daily retraining for a model that changes slowly (waste of compute)
- Match retraining frequency to drift rate

**Not testing in production-like environment:**
- Staging environment should mirror production (data, scale, dependencies)

## Best Practices

1. **Start simple:** Deploy a baseline model quickly, iterate
2. **Automate everything:** Training, testing, deployment (CI/CD for ML)
3. **Monitor early:** Don't wait for user complaints
4. **Version everything:** Code, data, models, features
5. **Plan for failure:** Fallbacks, circuit breakers, graceful degradation
6. **Communicate clearly:** Model performance to stakeholders, limitations to users
7. **Document decisions:** Why this model? Why this threshold? Why this frequency?

## Useful Metrics

**Serving Metrics:**
- p50, p95, p99 latency
- QPS (queries per second)
- Error rate (5xx, timeouts)
- Cost per query

**Model Metrics:**
- Online accuracy (from labeled sample)
- Prediction distribution (are predictions well-calibrated?)
- Confidence distribution (too confident? not confident enough?)

**Data Metrics:**
- Feature null rates (missing data increasing?)
- Feature distributions (KL divergence from training)
- Schema violations (unexpected types, out-of-range values)

**Business Metrics:**
- Revenue impact (A/B test)
- User engagement (CTR, time spent)
- Customer satisfaction (CSAT, NPS)
