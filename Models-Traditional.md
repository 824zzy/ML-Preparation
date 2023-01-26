# Machine Learning Models

## Naive Bayes

Bayes' theorem formula: $P(A|B) = P(B|A) * P(A) / P(B)$

The Naive Bayes method is a supervised learning algorithm, it is naive since it makes assumptions by applying Bayes’ theorem that all attributes are independent of each other.

## Principal component analysis (PCA)

Principal component analysis (PCA) is most commonly used for dimension reduction. 

Step 1: Standardize the data
Step 2: Compute the covariance matrix
Step 3: Compute eigen vectors of the covariance matrix
Step 4: Compute the explained variance and select N components
Step 5: Transform Data using eigen vectors
Step 6: Invert PCA and Reconstruct original data

## Support Vector Machine (SVM)

Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both **classification or regression challenges**.

A Support Vector Machine (SVM) is an algorithm that tries to **fit a line (or plane or hyperplane) between the different classes** that **maximizes the distance** from the line to the points of the classes.

What are the different kernels in SVM?

1. Linear kernel - used when data is linearly separable.
2. Polynomial kernel - When you have discrete data that has no natural notion of smoothness.
3. Radial basis kernel - Create a decision boundary able to do a much better job of separating two classes than the linear kernel.
4. Sigmoid kernel - used as an activation function for neural networks.

## Logistic Regression

Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable.

In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression).

## K-Nearest Neighbors(KNN)

KNN is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space.

1. Step1: Choose K value
2. Step2: For each data point in the data:
   1. Find the Euclidean distance to all training data samples
   2. Store the distances on an ordered list and sort it
   3. Choose the top K entries from the sorted list
3. Step3: Label the test point based on the majority of classes present in the selected points

## K-Means Clustering

1. Step 1: Determine K value by Elbow method and specify the number of clusters K
2. Step 2: Randomly assign each data point to a cluster
3. Step 3: Determine the cluster centroid coordinates
4. Step 4: Determine the distances of each data point to the centroids and re-assign each point to the closest cluster centroid based upon minimum distance
5. Step 5: Calculate cluster centroids again
6. Step 6: Repeat steps 4 and 5 until we reach global optima where no improvements are possible and no switching of data points from one cluster to other.

## Ensemble Learning

Ensemble is a machine learning concept, the basic idea is to learn a set of classifiers (experts) and to allow them to vote.

Bagging and Boosting are two types of Ensemble Learning.

1. What is bagging and boosting?
   1. Bagging is democracy based politics for less variance
      1. Step 1: Multiple subsets are created from the original data set with equal tuples, selecting observations with replacement.
      2. Step 2: A base model is created on each of these subsets.
      3. Step 3: Each model is learned in parallel with each training set and independent of each other.
      4. Step 4: The final predictions are determined by combining the predictions from all the models.
   2. Bagging is elite based politics for less bias
      1. Step 1: Initialize the dataset and assign equal weight to each of the data point.
      2. Step 2: Provide this as input to the model and identify the wrongly classified data points.
      3. Step 3: Increase the weight of the wrongly classified data points and decrease the weights of correctly classified data points. And then normalize the weights of all data points.
      4. Step 4: Loop until got required results.
2. What is Adaboost. How is it different from Adagrad

## Reference

- [A Step By Step Implementation of Principal Component Analysis](https://towardsdatascience.com/a-step-by-step-implementation-of-principal-component-analysis-5520cc6cd598)