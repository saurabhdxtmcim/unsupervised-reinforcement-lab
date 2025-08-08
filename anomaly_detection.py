# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os

"""
Anomaly Detection Algorithm for Server Computer Behavior

Problem Statement:
    Implement an anomaly detection algorithm to detect anomalous behavior in server computers.
    The goal is to identify servers that are behaving unusually compared to normal operation patterns.

Dataset Features:
    - throughput (mb/s): Data transfer rate of each server
    - latency (ms): Response time of each server
    - m = 307 examples of unlabeled data from server operation

Approach:
    1. Use a Gaussian model to detect anomalous examples in the dataset
    2. Start with a 2D dataset to visualize what the algorithm is doing
    3. Fit a Gaussian distribution to the data
    4. Identify values with very low probability as potential anomalies
    5. Apply the anomaly detection algorithm to larger datasets with many dimensions

Assumptions:
    - The vast majority of examples are "normal" (non-anomalous) server behavior
    - Some examples may represent servers acting anomalously within the dataset
    - Normal server behavior follows a Gaussian distribution pattern

Implementation Steps:
    - Data visualization and exploration
    - Gaussian parameter estimation (mean and covariance)
    - Probability calculation for each data point
    - Threshold selection for anomaly detection
    - Anomaly identification and evaluation
"""

def load_data():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    X = np.load(os.path.join(data_dir, "X_part1.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val_part1.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val_part1.npy"))
    return X, X_val, y_val

# load data
X_train, X_val, y_val = load_data()

print('first 5 elements of X_train: ', X_train[:5])
print('first 5 elements of X_val: ', X_val[:5])
print('first 5 elements of y_val: ', y_val[:5])

print('the shape of X_train is: ', X_train.shape)
print('the shape of X_val is: ', X_val.shape)
print('the shape of y_val is: ', y_val.shape)

# plot the data
plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', color='b', label='Normal')
plt.scatter(X_val[:, 0], X_val[:, 1], marker='o', color='r', label='Anomalous')
plt.legend()
plt.show()

"""
Gaussian Distribution Parameter Estimation

For anomaly detection, we need to fit a Gaussian model to the training data distribution.

Mathematical Foundation:
    The Gaussian distribution is given by:
    p(x; μ, σ²) = (1/√(2πσ²)) * exp(-(x-μ)²/2σ²)
    
    Where:
    - μ (mu) is the mean parameter
    - σ² (sigma squared) is the variance parameter

Parameter Estimation Process:
    For each feature i = 1...n, we need to estimate:
    - μᵢ: mean of the i-th feature across all training examples
    - σᵢ²: variance of the i-th feature across all training examples
    
    This allows us to model the probability distribution of normal behavior
    for each feature dimension independently.

Next Steps:
    1. Implement estimate_gaussian() function to calculate μ and σ² for each feature
    2. Calculate probability p(x) for each example using the Gaussian model
    3. Set threshold ε to classify examples as normal or anomalous
    4. Examples with p(x) < ε will be flagged as anomalies
"""

def estimate_gaussian(X):
    """
    Estimate the parameters of a Gaussian distribution using the dataset X
    
    Args:
        X (ndarray): (m, n) dataset, m examples with n features each
        
    Returns:
        mu (ndarray): (n,) mean of each feature
        sigma2 (ndarray): (n,) variance of each feature
    """
    m, n = X.shape
    mu = 1/m * np.sum(X, axis=0)
    variance = 1/m * np.sum((X - mu)**2, axis=0)
    return mu, variance

mu, variance = estimate_gaussian(X_train)

print('mean of each feature: ', mu)
print('variance of each feature: ', variance)

# plot the the guassian distribution
def multivariate_gaussian(X, mu, var):
    """
    Computes the probability 
    density function of the examples X under the multivariate gaussian 
    distribution with parameters mu and var. If var is a matrix, it is
    treated as the covariance matrix. If var is a vector, it is treated
    as the var values of the variances in each dimension (a diagonal
    covariance matrix
    """
    
    k = len(mu)
    
    if var.ndim == 1:
        var = np.diag(var)
        
    X = X - mu
    p = (2* np.pi)**(-k/2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))
    
    return p
    
def visualize_fit(X, mu, var):
    """
    This visualization shows you the 
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
    """
    
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariate_gaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, var)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx')

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), linewidths=1)
        
    # Set the title
    plt.title("The Gaussian contours of the distribution fit to the dataset")
    # Set the y-axis label
    plt.ylabel('Throughput (mb/s)')
    # Set the x-axis label
    plt.xlabel('Latency (ms)')

visualize_fit(X_train, mu, variance)
plt.show()

"""
Threshold Selection for Anomaly Detection (Section 2.3.2)

Now that we have estimated the Gaussian parameters (μ, σ²), we need to determine 
the threshold ε to classify examples as normal or anomalous.

Threshold Selection Process:
    - Examples with probability p(x) < ε are classified as anomalies
    - Examples with probability p(x) ≥ ε are classified as normal
    - We use cross-validation data (X_val, y_val) to find the optimal threshold

Cross-Validation Data:
    - X_val: validation examples with computed probabilities p(x_cv)
    - y_val: ground truth labels where y=1 means anomaly, y=0 means normal
    - This labeled data helps us evaluate different threshold values

F1 Score Optimization:
    The select_threshold function will:
    1. Try many different values of ε (threshold)
    2. For each ε, classify validation examples as anomalies if p(x) < ε
    3. Calculate metrics:
       - tp (true positives): correctly identified anomalies
       - fp (false positives): normal examples incorrectly flagged as anomalies  
       - fn (false negatives): anomalies missed by the algorithm
    4. Compute precision = tp/(tp + fp) and recall = tp/(tp + fn)
    5. Calculate F1 score = (2 * precision * recall)/(precision + recall)
    6. Select ε that maximizes F1 score

Next Implementation:
    - Implement select_threshold(y_val, p_val) function
    - Calculate probabilities p_val for validation data using multivariate_gaussian
    - Find optimal threshold using F1 score optimization
"""
def select_threshold(y_val, p_val):
    """
    Find the best threshold (epsilon) to use for selecting outliers based on the results from a validation set
    (p_val) and the ground truth (y_val).
    """
    
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        predictions = p_val < epsilon
        tp = np.sum((predictions == 1) & (y_val == 1))
        fp = np.sum((predictions == 1) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        print('f1: ', f1)
        print('epsilon: ', epsilon)
        print('tp: ', tp)
        print('fp: ', fp)
        print('fn: ', fn)
        print('precision: ', precision)
        print('recall: ', recall)
        print('--------------------------------')
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1

p_val = multivariate_gaussian(X_val, mu, variance)

best_epsilon, best_f1 = select_threshold(y_val, p_val)
print('best_epsilon: ', best_epsilon)
print('best_f1: ', best_f1)

# calculate probabilities for training set
p_train = multivariate_gaussian(X_train, mu, variance)

# find the outliers in the training set
outliers = p_train < best_epsilon

# visualize the outliers
# Visualize the fit
visualize_fit(X_train, mu, variance)
plt.scatter(X_train[outliers, 0], X_train[outliers, 1], marker='o', facecolor='none', edgecolor='r', label='Outliers')
plt.legend()
plt.show()

"""
High Dimensional Dataset

Now we apply the anomaly detection algorithm to a more realistic and challenging dataset.

Dataset Characteristics:
    - 11 features per example (instead of 2)
    - Captures many more properties of compute servers
    - More realistic representation of server monitoring data
    - Higher complexity requires the same Gaussian modeling approach

Data Variables (with _high suffix for distinction):
    - X_train_high: Training data with 11 features for Gaussian parameter estimation
    - X_val_high: Validation data with 11 features for threshold selection
    - y_val_high: Ground truth labels for validation (1=anomaly, 0=normal)

Implementation Process:
    1. Load the high-dimensional dataset using load_data() function
    2. Apply the same estimate_gaussian() function to X_train_high
    3. Calculate probabilities using multivariate_gaussian() for validation data
    4. Use select_threshold() to find optimal epsilon using F1 score
    5. Identify anomalies in the high-dimensional space
    6. Evaluate performance on this more challenging dataset

Key Difference:
    - Cannot visualize 11D data directly (no scatter plots)
    - Same mathematical approach: fit Gaussian, calculate probabilities, set threshold
    - Higher dimensionality may reveal more subtle anomaly patterns
"""

def load_data_multi():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    X = np.load(os.path.join(data_dir, "X_part2.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val_part2.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val_part2.npy"))
    return X, X_val, y_val

X_train_high, X_val_high, y_val_high = load_data_multi()

print('the shape of X_train_high is: ', X_train_high.shape)
print('the shape of X_val_high is: ', X_val_high.shape)
print('the shape of y_val_high is: ', y_val_high.shape)

"""
Anomaly Detection on High-Dimensional Dataset

Now applying the complete anomaly detection pipeline to the 11-feature dataset.

Pipeline Steps:
    1. Estimate Gaussian Parameters (μᵢ and σᵢ²):
       - Use estimate_gaussian() on X_train_high 
       - Calculate mean and variance for each of the 11 features
       - These parameters define the normal behavior distribution
       
    2. Evaluate Probabilities:
       - Calculate p(x) for training data X_train_high using multivariate_gaussian()
       - Calculate p(x) for validation data X_val_high using same parameters
       - Lower probabilities indicate more anomalous behavior
       
    3. Find Optimal Threshold:
       - Use select_threshold() with validation probabilities and ground truth labels
       - Optimize F1 score to balance precision and recall
       - Find threshold ε that best separates normal from anomalous examples

Expected Results:
    - More sophisticated anomaly detection with 11 features
    - Better capture of complex server behavior patterns
    - Higher dimensional space may reveal subtle anomalies missed in 2D analysis
"""

# Apply the same steps to the larger dataset

# Estimate the Gaussian parameters
mu_high, var_high = estimate_gaussian(X_train_high)

# Evaluate the probabilites for the training set
p_high = multivariate_gaussian(X_train_high, mu_high, var_high)

# Evaluate the probabilites for the cross validation set
p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)

# Find the best threshold
epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

print('Best epsilon found using cross-validation: %e'% epsilon_high)
print('Best F1 on Cross Validation Set:  %f'% F1_high)
print('# Anomalies found: %d'% sum(p_high < epsilon_high))

