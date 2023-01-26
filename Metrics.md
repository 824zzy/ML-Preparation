# Metrics for Evaluating Performance

## Distance Metrics

1. Euclidean Distance

![ed](2022-09-07-12-17-59.png)

2. Manhattan Distance

![MD](2022-09-07-12-18-24.png)

3. Minkowski Distance

![MD](2022-09-07-12-18-48.png)

4. Hamming Distance

## Evaluation Metrics

### Precision and Recall

Precision and recall are two metrics that are used to evaluate the performance of a classification model. They are defined as follows:

- **Precision**: The precision of a model is the ratio of the number of true positives to the total number of true positives and false positives. It is a measure of the model’s accuracy. A model with high precision is more likely to predict a positive class when it is actually positive.
- **Recall**: The recall of a model is the ratio of the number of true positives to the total number of true positives and false negatives. It is a measure of the model’s completeness. A model with high recall is more likely to predict a positive class when it is actually negative.

![PR](2022-09-07-12-08-30.png)

### F1 score

The F1-score of a classification model is calculated as the harmonic mean of the precision and recall of the model. 

It is a good measure to use if you have an uneven class distribution (i.e. a lot more positive samples than negative samples).

![f1](2022-09-07-12-06-25.png)

### ROC and AUC

The ROC curve is a plot of the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. The AUC is the area under the ROC curve.

AUC is the larger the better. AUC is useful as a single number summary of classifier performance.