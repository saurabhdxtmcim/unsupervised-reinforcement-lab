# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

"""
Collaborative Filtering Recommender Systems

Problem Statement:
    Implement a collaborative filtering learning algorithm for movie recommendations.
    The goal is to generate two types of vectors for each user and movie to predict ratings.

System Components:
    - User Parameter Vector w^(j): Embodies the movie tastes/preferences of user j
    - Movie Feature Vector x^(i): Embodies descriptive properties of movie i  
    - Bias Parameter b^(j): Additional bias term for user j
    - Prediction: rating = w^(j) · x^(i) + b^(j)

Mathematical Notation:
    - r(i,j): scalar, 1 if user j rated movie i, 0 otherwise
    - y(i,j): scalar, actual rating given by user j on movie i (if r(i,j) = 1)
    - w^(j): vector, learned parameters for user j
    - b^(j): scalar, learned bias parameter for user j  
    - x^(i): vector, learned feature ratings for movie i
    - n_u: number of users
    - n_m: number of movies  
    - n: number of features
    - X: matrix of vectors x^(i)
    - W: matrix of vectors w^(j)
    - R: matrix of elements r(i,j)

Learning Process:
    1. Initialize user parameter vectors W and movie feature vectors X randomly
    2. Use existing user/movie ratings as training data
    3. Learn parameters simultaneously using gradient descent on cost function
    4. Collaborative aspect: users collaborate to generate the rating set
    5. Feature vectors must satisfy all users while user vectors must satisfy all movies

Prediction Formula:
    For any user-movie pair: predicted_rating = w^(j) · x^(i) + b^(j)

Implementation Steps:
    1. Load and explore the movie ratings dataset
    2. Implement the collaborative filtering cost function (cofiCostFunc)
    3. Use TensorFlow custom training loop to learn parameters
    4. Generate predictions for unrated movies
    5. Evaluate and visualize recommendations

Section 3 - Movie Ratings Dataset:

Dataset Source:
    The data set is derived from the MovieLens "ml-latest-small" dataset.
    [F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: 
    History and Context. ACM Transactions on Interactive Intelligent Systems (TiIS) 5,
    4: 19:1-19:19. https://doi.org/10.1145/2827872]

Dataset Specifications:
    - Original dataset: 9000 movies rated by 600 users
    - Reduced dataset: Focus on movies from years since 2000
    - Final dimensions: n_u = 443 users, n_m = 4778 movies
    - Rating scale: 0.5 to 5 in 0.5 step increments

Matrix Structures:
    - Y (n_m x n_u matrix): Stores the ratings y^(i,j)
    - R (binary indicator matrix): R(i,j) = 1 if user j gave a rating to movie i, 
      R(i,j) = 0 otherwise

Parameter Matrices (with n = 10 features):
    - X: (n_m x 10) matrix where i-th row corresponds to feature vector x^(i) for i-th movie
    - W: (n_u x 10) matrix where j-th row corresponds to parameter vector w^(j) for j-th user  
    - b: (n_u,) vector where each element is bias parameter b^(j) for j-th user
    - Both x^(i) and w^(j) are n-dimensional vectors (10 elements each)
    - Correspondingly, X is a n_m x 10 matrix and W is a n_u x 10 matrix

Loading Strategy:
    - Load Y and R with the movie dataset
    - Load X, W, and b with pre-computed values initially
    - These values will be learned later in the lab, but use pre-computed values 
      to develop the cost model first
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
import pandas as pd
import os

def load_precalc_params_small():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    X = loadtxt(os.path.join(data_dir, 'small_movies_X.csv'), delimiter=",")
    W = loadtxt(os.path.join(data_dir, 'small_movies_W.csv'), delimiter=",")
    b = loadtxt(os.path.join(data_dir, 'small_movies_b.csv'), delimiter=",")
    b = b.reshape(1,-1)
    num_movies, num_features = X.shape
    num_users,_ = W.shape
    return(X, W, b, num_movies, num_features, num_users)

def load_ratings_small():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    Y = loadtxt(os.path.join(data_dir, 'small_movies_Y.csv'), delimiter=",")
    R = loadtxt(os.path.join(data_dir, 'small_movies_R.csv'), delimiter=",")
    return(Y, R)
    
#load data
X, W, b, num_movies, num_features, num_users = load_precalc_params_small()

Y, R = load_ratings_small()

#print the shape of Y and R
print(Y.shape)
print(R.shape)

#print the shape of X, W, and b
print("X shape:", X.shape)
print("W shape:", W.shape)
print("b shape:", b.shape)
print("num_movies:", num_movies)
print("num_users:", num_users)
print("num_features:", num_features)

# From the matrix we can compute statistics like the average rating
tsmean = np.mean(Y[0, R[0, :].astype(bool)])
print("Average rating for movie 1:", tsmean)

"""
Collaborative Filtering Learning Algorithm Implementation

Collaborative Filtering Cost Function (cofiCostFunc)

The collaborative filtering cost function is mathematically defined as:

J(x^(0),...,x^(nm-1), w^(0), b^(0),...,w^(nu-1), b^(nu-1)) = 
    [1/2 * Σ(i,j):r(i,j)=1 (w^(j)·x^(i) + b^(j) - y^(i,j))²] + 
    [λ/2 * Σ(j=0 to nu-1)Σ(k=0 to n-1)(w_k^(j))² + λ/2 * Σ(i=0 to nm-1)Σ(k=0 to n-1)(x_k^(i))²]

Cost Function Components:
    1. Prediction Error Term (First Bracket):
       - Sums over all (i,j) pairs where r(i,j) = 1 (user j rated movie i)
       - Computes squared error between predicted rating (w^(j)·x^(i) + b^(j)) and actual rating y^(i,j)
       - Factor of 1/2 for mathematical convenience in derivatives
    
    2. Regularization Term (Second Bracket):
       - Prevents overfitting by penalizing large parameter values
       - λ (lambda) controls regularization strength
       - Includes regularization for both user parameters W and movie features X
       - Does NOT regularize bias terms b^(j)

Alternative Summation Notation:
    The first summation can also be written as:
    = [1/2 * Σ(j=0 to nu-1)Σ(i=0 to nm-1) r(i,j) * (w^(j)·x^(i) + b^(j) - y^(i,j))²] + regularization

Implementation Strategy:
    - Use vectorized operations with NumPy/TensorFlow for efficiency
    - Leverage matrix operations: predictions = X @ W.T + b
    - Apply R matrix as mask to include only rated movies in cost calculation
    - Compute regularization terms using element-wise operations on W and X matrices
"""
def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Collaborative filtering cost function.
    
    Parameters:
        X: (n_m, n) matrix of movie features
        W: (n_u, n) matrix of user parameters
        b: (n_u,) vector of user biases
        Y: (n_m, n_u) matrix of ratings
        R: (n_m, n_u) binary matrix of rating indicators
        lambda_: regularization parameter
        
    Returns:
        cost: scalar cost
        grad_X: (n_m, n) matrix of partial derivatives with respect to X
        grad_W: (n_u, n) matrix of partial derivatives with respect to W
        grad_b: (n_u,) vector of partial derivatives with respect to b
    """
    # Number of users and movies
    nm, nu = Y.shape
    J = 0.0

    for j in range(nu):
        w = W[j, :]
        b_j = b[0, j]

        for i in range(nm):
            if R[i, j]:
                x = X[i, :]
                y = Y[i, j]
                pred = np.dot(w, x) + b_j
                J += (pred - y) ** 2

    J = J / 2
    J += (0.5 * lambda_) * (np.sum(W ** 2) + np.sum(X ** 2) + np.sum(b ** 2)) 
    print(J)
    return(J)


#reduce the data set size so that it runs faster
num_users_r = 4
num_movies_r = 5
num_features_r = 3

X_r = X[:num_movies_r, :num_features_r]
W_r = W[:num_users_r, :num_features_r]
b_r = b[0, :num_users_r].reshape(1, -1)
Y_r = Y[:num_movies_r, :num_users_r]
R_r = R[:num_movies_r, :num_users_r]

print(f"b_r shape: {b_r.shape}")
# Evaluate the cost function
J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 0)

print(f"cost : {J}")

# Evaluate the cost function with regularization
J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 1.5)
print(f"Cost (with regularization): {J}")

#------------------------------------------------------------------
"""
Vectorized Implementation

It is important to create a vectorized implementation to compute J, since it will later be called many times during optimization. The linear algebra utilized is not the focus of this series, so the implementation is provided. If you are an expert in linear algebra, feel free to create your version without referencing the code below.

Current Implementation Issues:
    - Uses nested loops which are inefficient for large datasets
    - Iterates through each user-movie pair individually
    - Will be slow when called repeatedly during gradient descent optimization

Vectorized Implementation Benefits:
    - Leverages NumPy's optimized matrix operations
    - Computes all predictions simultaneously using matrix multiplication
    - Significantly faster execution, especially for large datasets
    - Essential for practical machine learning optimization

Key Vectorization Concepts:
    1. Matrix Multiplication: predictions = X @ W.T + b
    2. Element-wise Operations: Use broadcasting for efficient computation
    3. Masking: Apply R matrix to include only rated movies in cost calculation
    4. Batch Processing: Process all users and movies simultaneously

Recommended Approach:
    - Replace nested loops with matrix operations
    - Use np.sum() with axis parameters for dimension reduction
    - Apply R as a mask: cost_term = R * (predictions - Y) ** 2
    - Leverage NumPy broadcasting for bias addition
"""

def cofi_cost_func_vec(X, W, b, Y, R, lambda_):
    """
    Collaborative filtering cost function.
    
    Parameters:
        X: (n_m, n) matrix of movie features
        W: (n_u, n) matrix of user parameters
        b: (n_u,) vector of user biases
        Y: (n_m, n_u) matrix of ratings
        R: (n_m, n_u) binary matrix of rating indicators
        lambda_: regularization parameter
    """
    J = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    J = 0.5 * tf.reduce_sum(J ** 2) + (lambda_ / 2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return(J)

# Evaluate the cost function
J = cofi_cost_func_vec(X_r, W_r, b_r, Y_r, R_r, 0)
print(f"cost : {J}")

# Evaluate the cost function with regularization
J = cofi_cost_func_vec(X_r, W_r, b_r, Y_r, R_r, 1.5)
print(f"Cost (with regularization): {J}")