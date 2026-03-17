# 01_Fundamentals

Core ML concepts that come up in domain/depth rounds. Know these cold before moving to LLM/GenAI or system design.

## When to Study What

Use this table to decide what to review based on interview signals.

| Interview Signal | Topics to Review | Time Needed |
|-----------------|------------------|-------------|
| "Tell me about bias-variance tradeoff" | Concepts.ipynb | 30 min |
| "When would you use SVM vs random forest?" | Models_Classical.ipynb | 1-2 hours |
| "Explain how attention works" | Layers.ipynb | 1 hour |
| "Why ReLU instead of sigmoid?" | Activation_Functions.ipynb | 30 min |
| "What loss function for imbalanced classification?" | Loss_Functions.ipynb | 1 hour |
| "Adam vs SGD, when to use each?" | Optimizers.ipynb | 30 min |
| "How do you evaluate a ranking model?" | Evaluation_Metrics.ipynb | 1 hour |
| "Walk me through training a neural network" | All notebooks | 3-4 hours |

## Notebooks in This Directory

### Concepts.ipynb
Foundational concepts that interviewers expect you to know:
- Bias-variance tradeoff (the classic question)
- Supervised vs unsupervised vs semi-supervised
- Overfitting and regularization techniques
- Curse of dimensionality
- Train/val/test split strategies
- Cross-validation

### Models_Classical.ipynb
Classical ML algorithms still tested for breadth:
- Linear regression, logistic regression
- Decision trees, random forests, gradient boosting
- SVM (kernels, margin, support vectors)
- KNN (curse of dimensionality trap)
- K-means clustering
- PCA and dimensionality reduction

When to use each, their assumptions, and tradeoffs.

### Layers.ipynb
Neural network building blocks:
- Attention mechanism (self-attention, multi-head, cross-attention)
- Dropout (why it works, where to place it)
- Batch normalization vs layer normalization
- Pooling layers (max, average, global)
- Convolution layers (filters, stride, padding)
- Residual connections

Focus on attention. It's asked in 60% of interviews now.

### Activation_Functions.ipynb
Why each activation exists and when to use it:
- Sigmoid (output layer for binary classification)
- Tanh (hidden layers, zero-centered)
- ReLU (default choice, dying ReLU problem)
- Leaky ReLU, PReLU, ELU (fixes for dying ReLU)
- Softmax (multi-class output layer)
- GELU (used in transformers)

### Loss_Functions.ipynb
Critical for system design + domain rounds:
- Cross-entropy (classification)
- MSE, MAE (regression)
- Hinge loss (SVM)
- Dice loss (segmentation)
- L1 and L2 regularization
- RLHF and DPO intro (for LLM roles)

Know when each is appropriate and how to handle class imbalance.

### Optimizers.ipynb
How models actually learn:
- SGD (baseline, momentum variant)
- RMSprop (adaptive learning rates)
- Adam (default choice, when it fails)
- Learning rate schedules (warmup, decay)
- Gradient clipping

### Evaluation_Metrics.ipynb
How to measure success:
- Classification: precision, recall, F1, ROC/AUC
- Ranking: NDCG, MAP, MRR
- Regression: RMSE, MAE, R²
- NLP: BLEU, ROUGE, perplexity
- When to use which metric (imbalanced data, ranking tasks, etc.)

## Key Concepts Checklist

You should be able to explain these in 2-3 minutes each. If you can't, review the relevant notebook.

**Must-Know (asked in 70%+ of interviews)**
- [ ] Bias-variance tradeoff with examples
- [ ] Why and how attention works
- [ ] Cross-entropy loss for classification
- [ ] Precision vs recall with imbalanced data example
- [ ] Overfitting: how to detect and fix
- [ ] Batch normalization purpose and placement
- [ ] Adam optimizer and when it fails
- [ ] ROC/AUC interpretation

**Should-Know (asked in 40-60% of interviews)**
- [ ] When to use tree-based models vs neural networks
- [ ] Dropout as regularization
- [ ] ReLU vs sigmoid vs tanh tradeoffs
- [ ] L1 vs L2 regularization
- [ ] K-fold cross-validation
- [ ] Gradient descent variants (SGD, momentum, Adam)
- [ ] PCA for dimensionality reduction
- [ ] F1 score and when it's misleading

**Nice-to-Know (differentiator for senior roles)**
- [ ] Multi-head attention details
- [ ] Layer norm vs batch norm
- [ ] Learning rate warmup and why it helps
- [ ] Dice loss for segmentation
- [ ] NDCG for ranking evaluation
- [ ] Residual connections and vanishing gradients
- [ ] GELU vs ReLU in transformers
- [ ] RLHF basics

## Study Strategy

1. **If you have < 1 week**: Focus on the Must-Know checklist. Skim notebooks, don't code.
2. **If you have 1-2 weeks**: Read all notebooks. Implement 2-3 algorithms from scratch (attention, backprop, Adam).
3. **If you have 2+ weeks**: Go deep. Implement everything in `04_ML_Coding/` and connect it back here.

## Common Interview Questions

**Conceptual**
- "Explain bias-variance tradeoff. Give an example."
- "When would you use a tree-based model vs a neural network?"
- "Your model is overfitting. What do you do?"
- "Walk me through how attention works."

**Tradeoffs**
- "ReLU vs sigmoid. When and why?"
- "Adam vs SGD. When does Adam fail?"
- "Precision vs recall. Which matters more for fraud detection?"
- "Batch norm vs layer norm. Where do you use each?"

**Practical**
- "You have an imbalanced dataset. What loss function and metrics?"
- "Your learning curve shows high bias. What do you do?"
- "How do you choose a learning rate?"
- "What's the difference between L1 and L2 regularization?"

Answers and explanations are in the notebooks.
