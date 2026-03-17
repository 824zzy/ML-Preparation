# A/B Testing for ML Systems

## Overview

A/B testing is the gold standard for evaluating ML systems in production. Offline metrics (AUC, F1) don't always translate to online business impact. This document covers experimental design, statistical analysis, and common pitfalls.

## 1. Experimental Design

### Randomization Unit

**User-level randomization (most common):**
- Each user is assigned to control or treatment
- Pros: Easy to implement, no interference
- Cons: Need many users for significance

**Session-level randomization:**
- Each session gets a random assignment
- Pros: More data points (faster experiments)
- Cons: Same user may see both versions (can confuse them)

**Query-level randomization:**
- Each query gets randomly assigned
- Pros: Maximum data (fastest to significance)
- Cons: High interference (user sees inconsistent results)

**Cluster-level randomization:**
- Assign groups (e.g., cities, schools) to treatment
- Needed when network effects exist
- Example: Uber driver surge pricing (can't randomize by driver, affects other drivers)

### Choosing the Right Unit
Ask: Is there interference between units?
- Search ranking: User-level (queries from same user should be consistent)
- Ads CTR: User-level (avoid confusing user with different ads)
- Fraud detection: Transaction-level is OK (independent events)

### Sample Size Calculation

**Formula (two-sample t-test):**
```
n = (z_α/2 + z_β)² × 2σ² / δ²

where:
- z_α/2 = critical value for significance level (1.96 for α=0.05)
- z_β = critical value for power (0.84 for β=0.20, i.e., 80% power)
- σ = population standard deviation
- δ = minimum detectable effect (MDE)
```

**Example:**
- Current CTR: 5%
- Want to detect: 0.25% absolute increase (5% → 5.25%, relative increase of 5%)
- Significance: α = 0.05
- Power: 80%
- Estimated σ ≈ 0.22 (for binary outcome)

Result: Need ~100K users per group.

**Key insight:** Smaller effect requires larger sample size. If you want to detect 0.1% change, you need 6x more data than for 0.25% change.

### Control vs Treatment Split
- **50/50 split:** Most common, maximizes statistical power
- **90/10 split:** Use if treatment is risky (limit exposure)
- **Multi-arm:** Multiple treatments vs control (A/B/C testing)

## 2. Metrics Selection

### Primary Metric
The one metric that decides success/failure.
- Should align with business goal (revenue, engagement)
- Must be measurable within experiment duration
- Should have low variance (or need larger sample size)

**Examples:**
- Search: CTR, time to first click, success rate
- Ads: revenue per user, CTR
- Recommendation: click rate, watch time
- Fraud: fraud caught rate, false positive rate

### Secondary Metrics
Additional metrics to understand the full picture.
- User satisfaction
- Latency
- Engagement depth

### Guardrail Metrics
Metrics that must not degrade.
- Page load time (don't slow down the site)
- Error rate (don't break things)
- User retention (don't hurt long-term)
- Revenue (don't kill the business)

### Leading vs Lagging Metrics
- **Leading:** Immediate (CTR, latency)
- **Lagging:** Delayed (7-day retention, LTV)

For short experiments, use leading metrics. Validate with lagging metrics post-launch.

## 3. Statistical Testing

### Hypothesis Testing Basics

**Null hypothesis (H₀):** No difference between control and treatment
**Alternative hypothesis (H₁):** Treatment is different from control

**Type I error (false positive, α):** Conclude there's a difference when there isn't
- Typically set α = 0.05 (5% chance of false positive)

**Type II error (false negative, β):** Conclude there's no difference when there is
- Typically set β = 0.20 (80% power = 1 - β)

**P-value:** Probability of seeing this result (or more extreme) if null hypothesis is true
- If p < α (e.g., 0.05), reject null hypothesis (declare significance)

### T-Test (Continuous Metrics)
For metrics like revenue per user, time spent.

```python
from scipy import stats

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(control_metric, treatment_metric)

if p_value < 0.05:
    print("Statistically significant")
```

### Z-Test (Binary Metrics)
For metrics like CTR (clicked or not).

```python
# Example: CTR test
control_ctr = clicks_control / impressions_control
treatment_ctr = clicks_treatment / impressions_treatment

pooled_ctr = (clicks_control + clicks_treatment) / (impressions_control + impressions_treatment)
se = np.sqrt(pooled_ctr * (1 - pooled_ctr) * (1/impressions_control + 1/impressions_treatment))

z_stat = (treatment_ctr - control_ctr) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
```

### Confidence Intervals
More informative than p-value alone.

```python
# 95% confidence interval for difference in means
diff = treatment_mean - control_mean
se = np.sqrt(treatment_var/n_treatment + control_var/n_control)
ci_lower = diff - 1.96 * se
ci_upper = diff + 1.96 * se

print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

**Interpretation:**
- If CI includes 0: Not statistically significant
- If CI is [0.02, 0.05]: We're 95% confident the true effect is between 2% and 5%

## 4. Common Pitfalls

### Pitfall 1: Peeking
Checking results before experiment is done and stopping early if significant.

**Why it's bad:**
- Inflates Type I error (false positive rate goes from 5% to 30%+)
- P-values naturally fluctuate, early significance may disappear

**Solution:**
- Pre-commit to sample size and duration
- If you must peek, use sequential testing (adjust α)

### Pitfall 2: Multiple Testing
Running many tests increases chance of false positive.

**Example:**
- Test 20 metrics at α = 0.05
- Expected false positives: 20 × 0.05 = 1 metric will be "significant" by chance

**Solution:**
- Bonferroni correction: Use α/n (e.g., 0.05/20 = 0.0025)
- Benjamini-Hochberg (FDR control, less conservative)
- Or: Designate one primary metric, others are secondary (no correction needed)

### Pitfall 3: Novelty Effect
New feature gets initial boost, but effect fades over time.

**Why it happens:**
- Users try new thing out of curiosity
- Excitement wears off after a week

**Solution:**
- Run experiment for 2-4 weeks (not just 2-3 days)
- Monitor metric over time (is effect stable?)
- Check effect on new vs returning users

### Pitfall 4: Simpson's Paradox
Treatment wins overall but loses in every segment.

**Example:**
- Overall: Treatment CTR = 5%, Control CTR = 4.5% (treatment wins)
- Mobile: Treatment CTR = 3%, Control CTR = 3.5% (treatment loses)
- Desktop: Treatment CTR = 7%, Control CTR = 8% (treatment loses)

**Why it happens:**
- Different proportions in control vs treatment (imbalanced randomization, or users self-select)

**Solution:**
- Stratified analysis (analyze by segment)
- Check randomization quality (are groups balanced?)

### Pitfall 5: Selection Bias
Only certain users enter the experiment.

**Example:**
- Experiment on "users who click search button"
- Treatment makes search button more prominent
- Treatment group has more users (different population)

**Solution:**
- Randomize at the earliest possible point (before user action)
- Intent-to-treat analysis (analyze all assigned users, not just those who engaged)

### Pitfall 6: Metric Dilution
Experiment only affects a small % of users, but you measure across all users.

**Example:**
- Change search ranking for queries with "buy" keyword (10% of queries)
- Measure overall CTR (includes 90% unaffected queries)
- Hard to detect effect (diluted by unaffected queries)

**Solution:**
- Filter to affected users/queries
- Use a more sensitive metric

### Pitfall 7: Network Effects
One user's experience affects another user.

**Example:**
- Uber: Surge pricing for drivers in treatment group
- Affects ride availability for control group (they see fewer drivers)

**Solution:**
- Cluster randomization (randomize by city, not by driver)
- Switchback experiments (alternate time periods)

## 5. Beyond Simple A/B Testing

### Multi-Armed Bandits
Adaptive experiment that shifts traffic to better-performing variant.

**How it works:**
- Start with equal traffic split (50/50)
- As data comes in, shift more traffic to winning variant
- Balances exploration (gather data) and exploitation (maximize metric)

**Algorithms:**
- Epsilon-greedy (explore with probability ε)
- Thompson Sampling (Bayesian approach)
- Upper Confidence Bound (UCB)

**When to use:**
- Long-running experiments (months)
- High cost of suboptimal variant
- Less interpretable than fixed A/B test

### Interleaving
Show results from both models interleaved in the same list.

**Example (search ranking):**
- Control ranks: [A, B, C, D, E]
- Treatment ranks: [A, C, E, B, D]
- Interleaved: [A, C, B, E, D] (alternate picks from each list)
- Winner: Whichever model's results get more clicks

**Pros:**
- More sensitive (same user sees both models)
- Faster to significance (fewer users needed)

**Cons:**
- Only works for ranking (not for other ML systems)
- Complex to implement

### Sequential Testing
Allows peeking without inflating Type I error.

**Methods:**
- Sequential Probability Ratio Test (SPRT)
- Always Valid Inference (AVI)

**Trade-off:**
- Can stop early if strong signal
- But requires more data if weak signal

## 6. Practical Considerations

### Experiment Duration

**Minimum duration:**
- At least 1 week (capture day-of-week effects)
- Preferably 2-4 weeks (capture novelty effect)

**Avoid:**
- Starting on Monday, ending on Friday (biased sample)
- Running during holidays (atypical behavior)

**Rule of thumb:**
- 1 week: Quick sanity check
- 2 weeks: Standard experiment
- 4+ weeks: Major change, check long-term effects

### Allocation Strategy

**Fixed allocation:**
- 50/50 split throughout experiment
- Simplest, most interpretable

**Staged rollout:**
- 5% → 25% → 50% → 100%
- Safer for risky changes
- Takes longer to reach conclusion

**Holdback:**
- Keep 1-5% of users in control permanently
- Measure long-term effect
- Common for major changes

### When NOT to A/B Test

**Too risky:**
- Legal/ethical issues (can't randomly harm users)
- Major rebranding (can't show inconsistent brand)

**Too slow:**
- Need instant decision (model is broken, roll back immediately)
- Extremely rare events (fraud on luxury items, need years to detect difference)

**Not worth it:**
- Minor change with no expected impact (waste of time)
- Exploratory feature (just ship and iterate)

## 7. Reporting and Interpretation

### What to Report

**Experiment setup:**
- Randomization unit
- Sample size (per group)
- Duration
- Hypothesis

**Results:**
- Metric value (control, treatment)
- Absolute difference
- Relative difference (% change)
- P-value
- Confidence interval

**Guardrails:**
- Did any guardrail metric regress?

**Segment analysis:**
- How did different user segments respond?

**Recommendation:**
- Ship, don't ship, iterate

### Interpreting Results

**P < 0.05, effect is positive:**
- Ship (but check guardrails and segment analysis)

**P < 0.05, effect is negative:**
- Don't ship (treatment is worse)

**P > 0.05:**
- No significant difference detected
- Could be: (1) no real effect, (2) effect is too small, (3) not enough data
- Decision: Don't ship (not worth the risk/effort)

**Directionally positive but not significant:**
- Consider: Is the point estimate meaningful?
- If effect is large (e.g., +10% revenue) but not significant, maybe run longer
- If effect is tiny (e.g., +0.1% CTR) and not significant, don't bother

## 8. Advanced Topics

### Variance Reduction

If metric has high variance, need more data. Can we reduce variance?

**CUPED (Controlled-experiment Using Pre-Experiment Data):**
- Use pre-experiment metric to adjust post-experiment metric
- Reduces variance by 30-50% (need fewer users)

**Stratification:**
- Split users into strata (e.g., by activity level)
- Analyze each stratum separately, then combine
- Reduces variance if strata are homogeneous

### Long-term Effects

A/B tests measure short-term impact. How to measure long-term?

**Cohort analysis:**
- Track users who joined during experiment
- Measure 30-day, 90-day retention
- Slower but more informative

**Quasi-experiments:**
- Natural experiments (feature launched in region A before region B)
- Difference-in-differences

### Heterogeneous Treatment Effects

Treatment may affect different users differently.

**Approaches:**
- Segment analysis (test on power users vs casual users)
- Causal forests (ML to predict treatment effect by user)
- Subgroup discovery (find which segments benefit most)

## Example: A/B Test for Search Ranking

**Setup:**
- Randomization: User-level
- Split: 50/50 (control vs new ranking model)
- Sample size: 200K users per group
- Duration: 2 weeks
- Primary metric: CTR (click-through rate)
- Secondary: Time to first click, success rate
- Guardrails: Latency, error rate

**Results:**
- Control CTR: 30.2%
- Treatment CTR: 30.8%
- Absolute difference: +0.6%
- Relative difference: +2.0%
- P-value: 0.003 (significant)
- 95% CI: [+0.2%, +1.0%]

**Segment analysis:**
- Mobile: +3% CTR (p=0.001)
- Desktop: +1% CTR (p=0.10, not significant)

**Guardrails:**
- Latency: p99 unchanged
- Error rate: No change

**Decision:** Ship. The new model improves CTR, especially on mobile, with no downside.
