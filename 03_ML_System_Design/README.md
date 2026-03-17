# ML System Design

## The 6-Step Framework

Use this for every system design interview:

1. **Problem Definition** - Clarify requirements, define scope, identify ML task type
2. **Metrics** - Offline (precision/recall/AUC), online (CTR/engagement/revenue), guardrails
3. **Data** - Sources, labeling, features, preprocessing pipeline
4. **Model** - Architecture choices, training strategy, candidate generation → ranking
5. **Serving** - Batch vs real-time, latency budget, caching, A/B testing
6. **Monitoring** - Data drift, model performance, feedback loops

## Case Studies in This Directory

- `Framework.md` - Full template with detailed breakdowns
- `Search_Ranking.md` - Query understanding, retrieval, learning-to-rank
- `Ads_Click_Prediction.md` - Ad auction, CTR prediction, calibration
- `Content_Moderation.md` - Multi-modal detection, trust & safety trade-offs
- `Chatbot_QA_System.md` - LLM-based RAG, guardrails, evaluation
- `Fraud_Detection.md` - Extreme imbalance, real-time scoring, false positive cost

## Common Trade-offs to Discuss

**Precision vs Recall**
- Fraud detection: prioritize precision (false alarms are costly)
- Content moderation: balance shifts based on harm type
- Search: recall matters more (show all relevant results)

**Latency vs Accuracy**
- Real-time serving: simpler models, aggressive caching
- Batch predictions: can use heavier models
- Multi-stage ranking: fast candidate generation + slower re-ranking

**Online vs Offline Metrics**
- Offline AUC ≠ online CTR gain
- Novelty effects in A/B tests
- Long-term user satisfaction vs short-term engagement

**Model Complexity vs Interpretability**
- Finance/healthcare: need explainability
- Ads: deep models win but calibration matters
- Content moderation: hybrid approach with human review

## What Interviewers Look For

1. **Structured thinking** - Follow the framework, don't jump to model architecture
2. **Clarifying questions** - Ask about scale, latency requirements, existing infrastructure
3. **Breadth over depth** - Cover all 6 steps, then drill into 1-2 areas
4. **Trade-off awareness** - Everything has costs, discuss them explicitly
5. **Production mindset** - Don't just design the model, design the system
6. **Handling ambiguity** - Make reasonable assumptions and state them

## Interview Strategy

- Spend 5-10 minutes on problem definition and metrics
- The model step is important but shouldn't dominate (20-30% of time)
- Serving and monitoring often separate strong candidates from weak ones
- If you don't know something, say so and reason from first principles
- Draw diagrams for data flow and system architecture
