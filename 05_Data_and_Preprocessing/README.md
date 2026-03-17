# 05_Data_and_Preprocessing

Data work is 70% of real ML. Interviewers test this through system design questions and coding rounds. Know how to handle messy data, engineer features, and choose embeddings.

## When to Apply Each Technique

| Problem Signal | Technique | Notebook/Doc |
|---------------|-----------|--------------|
| "High cardinality categorical feature" | Target encoding, embeddings | Feature_Engineering.ipynb |
| "Text needs to be vectorized" | TF-IDF, word2vec, transformers | NLP_Preprocessing.ipynb, Embeddings.md |
| "Features have different scales" | Normalization, standardization | Feature_Engineering.ipynb |
| "Too many features, model is slow" | PCA, feature selection | Feature_Engineering.ipynb |
| "Missing values in dataset" | Imputation strategies | Data_Quality.md |
| "Need semantic search" | Dense embeddings (BERT, sentence transformers) | Embeddings.md |
| "Need to find similar items" | Embeddings + nearest neighbor search | Embeddings.md |
| "Class imbalance problem" | SMOTE, class weights, resampling | Data_Quality.md |
| "Data has outliers" | Clipping, robust scaling, binning | Feature_Engineering.ipynb |
| "Time series with missing gaps" | Forward fill, interpolation | Data_Quality.md |

## Notebooks and Docs

### Feature_Engineering.ipynb (Existing)
Covers feature creation and transformation:
- Categorical encoding (one-hot, label, target, frequency)
- Numerical transformations (log, sqrt, binning)
- Scaling (standardization, normalization, robust scaling)
- Feature interactions (polynomial, domain-specific)
- Dimensionality reduction (PCA, feature selection)
- Handling outliers

Focus on when to use each technique and their tradeoffs.

### NLP_Preprocessing.ipynb (Existing)
Text-specific preprocessing:
- Tokenization (word, subword, character)
- Cleaning (lowercasing, punctuation, stopwords)
- Stemming vs lemmatization
- TF-IDF vectorization
- N-grams
- When to skip preprocessing (modern transformers)

### Data_Quality.md (Planned)
Handling real-world messy data:
- Missing value strategies (mean/median/mode, KNN imputation, forward fill)
- Outlier detection and handling
- Class imbalance (SMOTE, class weights, undersampling, oversampling)
- Data validation checks
- Train/test leakage prevention
- Data versioning basics

### Embeddings.md (Planned)
Modern representation learning:
- When to use sparse vs dense embeddings
- Word embeddings (word2vec, GloVe, FastText)
- Contextual embeddings (BERT, RoBERTa)
- Sentence embeddings (Sentence-BERT, all-MiniLM)
- Image embeddings (ResNet, CLIP)
- How to evaluate embedding quality
- Approximate nearest neighbor search (FAISS, Annoy)

## Key Concepts Checklist

**Must-Know (comes up in system design + coding)**
- [ ] One-hot vs target encoding for categorical features
- [ ] When to normalize vs standardize
- [ ] Handling missing values (when to drop, impute, or model)
- [ ] Class imbalance strategies
- [ ] TF-IDF vs word embeddings
- [ ] How to prevent train/test leakage
- [ ] Feature scaling and why it matters for some models

**Should-Know (differentiator for design rounds)**
- [ ] High cardinality categorical features (target encoding, embeddings)
- [ ] Feature selection techniques (variance, correlation, model-based)
- [ ] SMOTE and when it helps vs hurts
- [ ] Sentence embeddings for semantic search
- [ ] Approximate nearest neighbor search
- [ ] Time series specific preprocessing
- [ ] Outlier detection methods

**Nice-to-Know (senior level)**
- [ ] Embedding fine-tuning strategies
- [ ] Data augmentation techniques by domain
- [ ] Feature stores and when to use them
- [ ] Online vs offline feature computation
- [ ] Cold start problem for embeddings
- [ ] Dimensionality reduction tradeoffs (PCA vs UMAP vs autoencoders)

## Common Interview Questions

**Feature Engineering**
- "You have a categorical feature with 10k unique values. What do you do?"
- "When would you use target encoding vs one-hot encoding?"
- "How do you handle missing values in production?"
- "Your numerical features have very different ranges. Does it matter?"

**NLP/Embeddings**
- "TF-IDF vs word2vec. When to use each?"
- "How would you build a semantic search system?"
- "Explain how sentence embeddings work."
- "Your embedding model is too slow at inference. What do you do?"

**Data Quality**
- "You have a 1:100 class imbalance. What do you do?"
- "How do you detect outliers? When do you remove them?"
- "Your validation accuracy is much lower than training. What could cause this?"
- "How do you prevent data leakage?"

**System Design Context**
- "Design a feature pipeline for a recommendation system."
- "How would you store and serve embeddings for 1B items?"
- "You need to add a new feature to production. Walk me through the process."

## Study Strategy

1. **System design prep**: Focus on Must-Know checklist. Be able to discuss tradeoffs.
2. **Coding prep**: Implement 2-3 preprocessing techniques from scratch (TF-IDF, target encoding, PCA).
3. **Domain prep**: Read through notebooks, focus on when to use each technique.

Most mistakes in interviews: choosing the wrong encoding for categorical features, not handling class imbalance, and causing train/test leakage. Make sure you know how to avoid these.

## Links to Other Topics

- Feature engineering connects to `01_Fundamentals/Models_Classical.ipynb` (which models need scaling?)
- Embeddings connect to `02_LLM_and_GenAI/` (how transformers create representations)
- Data quality connects to `03_ML_System_Design/` (how to build robust pipelines)
- All of this comes up in `04_ML_Coding/` (implement preprocessing in code)
