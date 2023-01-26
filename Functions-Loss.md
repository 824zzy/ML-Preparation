# 2. Loss Functions

## Dice Loss

The Dice coefficient, or Dice-Sørensen coefficient, is a common metric for pixel segmentation that can also be modified to act as a loss function.
Dice loss: $1 - \frac{2|X \cap Y|}{|X| + |Y|}$

## Cross Entropy Loss

Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. A perfect model would have a log loss of 0.

## MSE Loss (L2 Loss)

MSE Loss: $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2$.

Mean Squared Error as the name suggests is the sum of squared distances between our target variable and predicted values.

## MAE Loss (L1 Loss)

MAE Loss: $\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y_i}|$.

Mean Absolute Error is the sum of absolute distances between our target variable and predicted values.

## References

- [https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks](https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks)
- [https://towardsdatascience.com/fantastic-activation-functions-and-when-to-use-them-481fe2bb2bde](https://towardsdatascience.com/fantastic-activation-functions-and-when-to-use-them-481fe2bb2bde)