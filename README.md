# ML Interview Preparation

A comprehensive guide covering ML fundamentals through LLM/GenAI, system design, coding, and behavioral interviews. Built for preparing for MLE, Applied Scientist, and Research Engineer roles at top tech companies.

## Repository Structure

```
ML-Preparation/
├── 01_Fundamentals/           # Core ML concepts, models, layers, loss functions
├── 02_LLM_and_GenAI/          # Transformers, RLHF, prompt engineering, RAG
├── 03_ML_System_Design/       # Design problems, tradeoffs, case studies
├── 04_ML_Coding/              # Implement models/algorithms from scratch
├── 05_Data_and_Preprocessing/ # Feature engineering, data quality, embeddings
├── 06_MLOps_and_Deployment/   # Monitoring, A/B testing, production ML
├── 07_Behavioral/             # STAR stories, leadership principles
└── Company_Guides/            # Company-specific interview formats
```

## Study Roadmap

Prioritized by what's actually tested in 2026 interviews. Start with Priority 1.

| Priority | Topic | Why | Time Investment |
|----------|-------|-----|-----------------|
| 1 | **LLM/GenAI Fundamentals** | 80% of roles now test this. Know transformers, attention, RLHF, prompt engineering. | 2-3 weeks |
| 2 | **ML System Design** | The differentiator round. Design ranking systems, recommenders, search. | 2-3 weeks |
| 3 | **ML Coding** | Implement layers, training loops, simple models from scratch. Less common but high signal. | 1-2 weeks |
| 4 | **Classical ML Fundamentals** | Still tested for breadth. Know bias-variance, trees, SVMs, evaluation metrics. | 1 week |
| 5 | **Data/Preprocessing** | Comes up in system design + coding. Know feature engineering, embeddings, data quality. | 1 week |
| 6 | **MLOps/Deployment** | Tested for senior roles. Monitoring, A/B testing, model serving. | 1 week |
| 7 | **Behavioral** | Have 5-7 STAR stories ready. Map to company values. | 3-5 days |

**Total prep time**: 6-10 weeks depending on background.

## Company-Specific Guides

| Company | Interview Format | What They Care About | Notes |
|---------|------------------|---------------------|-------|
| **Meta** | 2 coding, 1 system design, 1 ML system design, 1 behavioral | Ranking/recommendation systems, scale, product sense | ML system design is the make-or-break round |
| **Google** | 1-2 coding, 1 ML domain, 1 ML system design, 1 behavioral | Scale, efficiency, breadth across ML topics | ML domain = deep dive on models, metrics, tradeoffs |
| **Anthropic** | System design (50-55 min), infra-focused, AI safety in every round | Alignment, safety, scalability, model training at scale | Expect safety questions woven into technical rounds |
| **OpenAI** | 5-7 rounds: pair coding/take-home, 4-6 hour final loop | Research taste, coding ability, alignment with mission | Final loop includes research presentation + deep dives |
| **Amazon** | 2 coding, 1 system design, 1 ML depth, 1-2 behavioral | Leadership principles, scale, pragmatism | Half the evaluation is behavioral (LPs) |
| **Apple** | 2-3 technical, 1 system design, 1 behavioral | On-device ML, privacy, efficiency | Focus on model compression, federated learning |

See `Company_Guides/` for detailed breakdowns and example questions.

## Interview Round Types

| Round Type | What's Tested | Study From | Duration |
|------------|---------------|------------|----------|
| **ML Coding** | Implement backprop, attention, loss functions from scratch | `04_ML_Coding/` | 45-60 min |
| **ML System Design** | Design ranking/search/recommender/fraud detection end-to-end | `03_ML_System_Design/` | 45-60 min |
| **Algorithms Coding** | LeetCode medium/hard, arrays, trees, graphs | External (LeetCode) | 45-60 min |
| **ML Domain/Depth** | Deep dive on models, metrics, tradeoffs. Expect follow-ups. | `01_Fundamentals/`, `02_LLM_and_GenAI/` | 45-60 min |
| **Behavioral** | STAR stories, leadership principles, past projects | `07_Behavioral/` | 30-45 min |
| **Take-Home** | Implement + evaluate model on dataset, write report | All topics | 3-6 hours |

## Key Resources

This repo is built on top of these excellent resources:

- [alirezadir/Machine-Learning-Interviews](https://github.com/alirezadir/Machine-Learning-Interviews) (comprehensive question bank)
- [khangich/machine-learning-interview](https://github.com/khangich/machine-learning-interview) (company-specific guides)
- [chiphuyen/ml-interviews-book](https://github.com/chiphuyen/ml-interviews-book) (great for system design)
- [eugeneyan/applied-ml](https://github.com/eugeneyan/applied-ml) (real-world ML systems)
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (learn transformers by implementing them)
- [Machine Learning Interview Questions](https://github.com/andrewekhalel/MLQuestions)
- [AmbitionBox](https://www.ambitionbox.com/interviews/jp-morgan-chase-interview-questions/machine-learning-engineer)
- [Top 45 Machine Learning Interview Questions](https://www.simplilearn.com/tutorials/machine-learning-tutorial/machine-learning-interview-questions)

## How to Use This Repo

1. Start with `02_LLM_and_GenAI/` if you're rusty. This is tested in 80% of roles now.
2. Move to `03_ML_System_Design/` and practice out loud. Whiteboard or use Excalidraw.
3. Brush up on `01_Fundamentals/` for domain rounds. Focus on the concepts checklist.
4. If coding rounds are confirmed, drill `04_ML_Coding/` and LeetCode in parallel.
5. Prep `07_Behavioral/` in the last week. Write down your STAR stories.

Good luck!
