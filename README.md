﻿# MetaCost-Cost-sensitive-learning-approach
1. What is Cost-sensitive learning?
**Definition**:

- A type of data mining, consider misclassification costs to enhance high accuracy
- For: imbalanced data

**Why need cost-sensitive learning**: 

- Standard classification algorithms such as decision trees, random forests, etc., focus on minimizing the overall error rate. They primarily aim to classify features into correct class labels, but they ignore differences between types of misclassification costs, assuming that all errors are equally costly. This often results in high false negatives, particularly in imbalanced datasets.
- Cost-sensitive learning incorporates misclassification costs into the classification process to address this limitation, leading to more balanced performance across classes.

2. How to implement MetaCost?

**Theory**: 
$$R(i|x) = \sum P(j|x)C(i,j)$$

R(i|x) is the expected cost of classifying x to class i

P(j|x) is the probability of classifying x to class j 

C(i,j) is the misclassification cost of predicting x, which belóng to true class i, is class j 

Assume that C(0,0) = C(1,1) = 1

**Algorithm**: 
1. Inputs
- S is the training set
- L is the learning algorithm
- C is the misclassification cost
- m is the number of resamples
- n is the number of examples of each resamples
- p is the boolean value represent for checking whether L produces distribution probability when applying L to S
- q is the boolean value stand for checking whether all resamples utilize x examples2.
2. Pseudocode
```pseudocode
for i in range(m):
    let S[i] be a resample generated from S
    let M[i] be the model produced by applying L to S[i]

for each example x in S:
    for each class j:
        let P[j|x] be the predicted probability from model M[i]

    Compute R(i|x) for each class i
    Update the label of x to argmin(R(i|x))

Retrain the model with the updated labels
```
