# Content Moderation System

Design a system to detect and filter harmful content (hate speech, spam, NSFW, violence, etc.) on a social platform.

## 1. Problem Definition

### Clarifying Questions
- What platform? (text-only like Twitter, images like Instagram, video like TikTok)
- Scale? (posts per day, MAU)
- Latency requirements? (pre-publish vs post-publish moderation)
- Policy scope? (what's considered harmful?)
- Consequences? (remove, downrank, warn user, ban account)

### Scope
- Focus: Detect harmful content before or after it goes live
- In scope: Multi-modal detection (text, image, video), human review pipeline
- Out of scope: User appeals process, localization (unless asked)

### Trust & Safety Context
This is a high-stakes problem. Errors have real consequences:
- **False negatives (miss harmful content):** Platform harm, legal risk, user safety
- **False positives (flag safe content):** User frustration, free speech concerns

The precision/recall trade-off depends on content type.

## 2. Metrics

### Offline Metrics
- **Precision:** Of flagged content, how much is truly harmful?
- **Recall:** Of all harmful content, how much did we catch?
- **F1 score:** Harmonic mean of precision and recall
- **AUC-ROC:** Overall classification quality
- **Per-category metrics:** Different thresholds for different harm types

### Online Metrics
- **Prevalence:** % of views that are harmful content (goal: <0.05%)
- **Actioned content rate:** % of posts flagged by model
- **Proactive detection rate:** % of harmful content caught before user report
- **Human review backlog:** Queue size for manual review
- **User reports:** Are users still seeing harmful content?

### Guardrail Metrics
- **Latency:** Moderation check < 200ms (don't slow down posting)
- **False positive rate on safe content:** Don't over-moderate
- **Demographic fairness:** Avoid bias against certain groups (dialect, race)
- **Transparency:** Can explain to users why content was removed

### Key Principle
For severe harms (CSAM, terrorism), optimize recall (miss nothing, even if false positives high). For borderline content (clickbait, spam), optimize precision (don't annoy users).

## 3. Data

### Data Sources

**User-generated content:**
- Text (posts, comments, messages)
- Images (photos, memes)
- Video (short clips, live streams)
- Metadata (timestamp, user history, engagement)

**User reports:**
- Explicit reports ("this is spam")
- Implicit signals (many users hide/block this)

**External signals:**
- Known bad actors (databases of CSAM hashes, terrorist content)
- Third-party fact-checkers (for misinformation)

### Labeling

**Human annotation:**
- Hire content moderators (traumatic work, need support)
- Guidelines: Detailed policy on what's violating
- Quality control: Multiple raters, measure inter-rater agreement
- Edge cases: Create taxonomy (hate speech vs criticism)

**Weakly supervised labels:**
- User reports (noisy but scalable)
- Historical takedowns (content that was removed)
- Heuristics (profanity list, known bad URLs)

**Active learning:**
- Model flags uncertain examples for human review
- Humans label, retrain model
- Iterative improvement

### Handling Class Imbalance
Harmful content is rare (<0.1% of posts typically).

**Solutions:**
- Oversample positives (duplicate harmful examples)
- Undersample negatives (but keep hard negatives)
- Class weights in loss function
- Focal loss (focus on hard examples)

### Features

**Text features:**
- Bag of words, TF-IDF
- Embeddings (BERT, RoBERTa)
- Linguistic features (profanity, all-caps, exclamation marks)
- Entity recognition (hate symbols, slurs)

**Image features:**
- CNN embeddings (ResNet, EfficientNet)
- Object detection (weapons, nudity)
- OCR + text moderation (memes with text)
- Hash matching (PhotoDNA for known CSAM)

**Video features:**
- Frame-level image analysis (sample frames)
- Audio transcription + text moderation
- Temporal patterns (violence detection)

**User features:**
- Historical violation rate
- Account age (new accounts more likely spam)
- Follower/following ratio
- Past reports/blocks

**Context features:**
- Time of day (spam surges at certain times)
- Geographic region (different norms)
- Engagement patterns (viral spread of misinformation)

## 4. Model

### Multi-Stage Pipeline

**Stage 1: Pre-filters (Fast Heuristics)**
- Hash matching (known bad content, exact duplicates)
- Profanity lists (keyword blocking)
- URL blacklists (phishing, malware)
- Rate limiting (same user posting 100 times/min)

Goal: Catch obvious violations instantly (<10ms).

**Stage 2: ML Classifiers (Medium Speed)**

**Text moderation:**
- BERT-based classifier (RoBERTa, DeBERTa)
- Multi-label: hate speech, spam, harassment, misinformation
- Calibrated threshold per category

**Image moderation:**
- CNN classifier (ResNet-50)
- Categories: NSFW, violence, hate symbols
- OCR + text classifier for memes

**Video moderation:**
- Sample frames (1 frame/sec), run image classifier
- Audio transcription (Whisper, then text classifier)
- Expensive, only for high-risk content

**Multi-modal fusion:**
- Some violations need context (image + caption)
- Late fusion: Combine scores from text and image models
- Early fusion: Transformer over concatenated embeddings

**Stage 3: Human Review (Slow but Accurate)**
- Model flags uncertain cases (confidence 0.4-0.6)
- Human moderators make final decision
- High-severity content (CSAM, terrorism) reviewed by specialized teams

### Model Architecture

**Text classifier:**
- Fine-tune RoBERTa on labeled data
- Output: Multi-label (can be spam AND hateful)
- Loss: Binary cross-entropy per label

**Image classifier:**
- Fine-tune EfficientNet or ResNet
- Output: Multi-label
- Data augmentation (rotation, crop) to increase robustness

**Ensemble:**
- Combine multiple models (reduces variance)
- Average probabilities or use stacking

### Calibration
Critical for setting thresholds.
- Platt scaling (fit sigmoid on validation set)
- Isotonic regression (non-parametric calibration)

### Adversarial Robustness
Bad actors try to evade detection:
- Misspelling ("h@te" instead of "hate")
- Character substitution (Cyrillic lookalikes)
- Adversarial images (pixel perturbations)

**Defenses:**
- Data augmentation during training (add synthetic perturbations)
- Character normalization (map lookalikes to ASCII)
- Adversarial training (add adversarial examples to training set)

## 5. Serving

### Pre-publish vs Post-publish

**Pre-publish (Proactive):**
- Check content before it goes live
- Pros: Prevent harm, better user experience
- Cons: Adds latency to posting, may slow down UX

**Post-publish (Reactive):**
- Content goes live immediately, flagged afterward
- Pros: No latency impact
- Cons: Harmful content visible until removed

**Hybrid (Most Common):**
- High-risk users: Pre-publish check
- Trusted users: Post-publish check
- Adjust based on user reputation

### Latency Budget
- Pre-publish check: <200ms (don't slow down posting too much)
- Text classifier: 50ms
- Image classifier: 100ms (can run in parallel with text)
- Video classifier: Too slow for pre-publish (run post-publish)

### Serving Architecture
```
User posts content
  → Fast pre-filters (hash, profanity)
  → ML classifiers (text, image)
  → Compute risk score (combine text + image + user history)
  → If high risk: Block or send to human review
  → If medium risk: Publish but flag for post-review
  → If low risk: Publish immediately
```

### Optimization Strategies

**Caching:**
- Cache embeddings for popular content (reposts)
- Cache user reputation scores

**Batching:**
- For post-publish, batch predictions (process 100 posts at once)

**Model optimization:**
- Quantization, pruning for faster inference
- Distillation (compress BERT into smaller model)

**Multi-stage ranking:**
- Cheap model (logistic regression) filters 90% as safe
- Expensive model (BERT) focuses on uncertain 10%

### Human-in-the-Loop
- Model flags uncertain cases (0.4 < confidence < 0.6)
- Humans review and make final decision
- Decisions feed back into training data
- Reduces false positives on edge cases

## 6. Monitoring

### Data Monitoring

**Content distribution drift:**
- New slang, memes, trends (model may not recognize)
- Seasonal events (political elections, holidays)
- Adversarial evolution (bad actors learn to evade)

**Label drift:**
- Policy changes (what's considered harmful evolves)
- Need to relabel old data with new policy

**Feature drift:**
- User behavior changes (more video, less text)
- Platform changes (new content types)

### Model Monitoring

**Performance by category:**
- Track precision/recall per harm type
- Hate speech model may degrade faster than spam model

**Performance by demographic:**
- Check for bias (over-flagging certain dialects, regions)
- Use fairness metrics (demographic parity, equalized odds)

**False positive analysis:**
- Sample flagged content and manually review
- Identify patterns (satire, educational content being flagged)

**False negative analysis:**
- Sample user-reported content that model missed
- Identify gaps (new attack vectors, edge cases)

### Online Metrics

**Prevalence:**
- % of impressions that are violating content
- Track over time, alert if spike

**Proactive rate:**
- % of violating content caught by model (vs user report)
- Goal: >95% proactive detection

**Human review metrics:**
- Queue depth (backlog of content awaiting review)
- Reviewer agreement with model (are we flagging the right stuff?)

### Feedback Loops

**Selection bias:**
- Model trains on flagged content (by model or users)
- May miss emerging patterns (if model never flags them, they never get labeled)

**Mitigation:**
- Random sampling of content for labeling (not just flagged content)
- Diversity in training data

**Over-moderation spiral:**
- Model flags borderline content, users complain less, model thinks it's working
- Reality: Users leave platform

**Mitigation:**
- Track user retention, engagement
- Regularly review false positives

### Retraining Strategy
- **Frequency:** Weekly (new slang, evasion tactics)
- **Active learning:** Prioritize labeling of uncertain predictions
- **Policy updates:** When policy changes, relabel data and retrain
- **Online learning:** Risky for moderation (one bad update can cause harm)

## Key Trade-offs

**Precision vs Recall:**
- High recall = catch all bad content, but many false positives
- High precision = minimize false positives, but miss some bad content
- **Recommendation:** Tune per category. CSAM, terrorism = maximize recall. Spam = balance. Borderline = favor precision.

**Speed vs Accuracy:**
- BERT is accurate but slow (100ms)
- Logistic regression is fast but less accurate (10ms)
- **Solution:** Multi-stage (fast filter → accurate classifier)

**Automation vs Human Review:**
- Full automation = scalable but error-prone
- Human review = accurate but slow and expensive
- **Solution:** Hybrid (automate obvious cases, humans review uncertain)

**Global vs Local Policies:**
- Global policy = easier to enforce, but may not fit all cultures
- Local policy = culturally appropriate, but complex to maintain
- **Solution:** Core global policy + regional overrides

## Common Interview Follow-ups

**"How do you handle satire or educational content?"**
- Context matters (quote discussing hate speech vs using hate speech)
- Add context features (user is journalist, educational account)
- Use human review for ambiguous cases
- Allow appeals

**"How do you handle multiple languages?"**
- Multilingual models (mBERT, XLM-RoBERTa)
- Language-specific models for high-volume languages
- Translation (translate to English, then classify)

**"How do you prevent adversarial attacks?"**
- Character normalization (map lookalikes)
- Adversarial training (add perturbed examples)
- Continuous monitoring (detect new evasion tactics)
- Human review for suspicious patterns

**"How do you explain decisions to users?"**
- Highlight problematic text/region in image
- Provide policy link
- Allow appeals with human review

**"How do you balance free speech and safety?"**
- This is a policy question, not just technical
- Discuss trade-offs explicitly
- Some platforms favor free speech (Twitter), others favor safety (Facebook)
- Technical system should be flexible to different policy choices

**"How do you handle misinformation?"**
- Harder than other harms (subjective, context-dependent)
- Combine model with fact-checkers (third-party partnerships)
- Downrank rather than remove (softer intervention)
- Add warning labels
