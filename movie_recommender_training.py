# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

import numpy as np
from numpy import loadtxt
import tensorflow as tf
from collaborative_recommender import *
import pandas as pd
import os

# helper functions--------------------------------
def load_Movie_List_pd():
    """ returns df with and index of movies in the order they are in in the Y matrix """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    df = pd.read_csv(os.path.join(data_dir, 'small_movie_list.csv'), header=0, index_col=0,  delimiter=',', quotechar='"')
    mlist = df["title"].to_list()
    return(mlist, df)

def load_ratings_small():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    Y = loadtxt(os.path.join(data_dir, 'small_movies_Y.csv'), delimiter=",")
    R = loadtxt(os.path.join(data_dir, 'small_movies_R.csv'), delimiter=",")
    return(Y,R)

def normalizeRatings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).
    Only include real ratings R(i,j)=1.
    [Ynorm, Ymean] = normalizeRatings(Y, R) normalized Y so that each movie
    has a rating of 0 on average. Unrated moves then have a mean rating (0)
    Returns the mean rating in Ymean.
    """
    Ymean = (np.sum(Y*R,axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R) 
    return(Ynorm, Ymean)

#main--------------------------------

#load movie list
movie_list, movieList_df = load_Movie_List_pd()

my_ratings = np.zeros(len(movie_list))
# Check the file small_movie_list.csv for id of each movie in our dataset
# For example, Toy Story 3 (2010) has ID 2700, so to rate it "5", you can set
my_ratings[2700] = 5 

#Or suppose you did not enjoy Persuasion (2007), you can set
my_ratings[2609] = 2;

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[929]  = 5   # Lord of the Rings: The Return of the King, The
my_ratings[246]  = 5   # Shrek (2001)
my_ratings[2716] = 3   # Inception
my_ratings[1150] = 5   # Incredibles, The (2004)
my_ratings[382]  = 2   # Amelie (Fabuleux destin d'Amélie Poulain, Le)
my_ratings[366]  = 5   # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
my_ratings[622]  = 5   # Harry Potter and the Chamber of Secrets (2002)
my_ratings[988]  = 3   # Eternal Sunshine of the Spotless Mind (2004)
my_ratings[2925] = 1   # Louis Theroux: Law & Disorder (2008)
my_ratings[2937] = 1   # Nothing to Declare (Rien à déclarer)
my_ratings[793]  = 5   # Pirates of the Caribbean: The Curse of the Black Pearl (2003)

print(my_ratings)

for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Rated {my_ratings[i]} for  {movieList_df.loc[i,"title"]}')

Y, R  = load_ratings_small()
# add new user ratings to Y
Y = np.c_[my_ratings, Y]

# add new user indicator matrix to R
R = np.c_[(my_ratings != 0).astype(int), R]

# Normalize Ratings
Ynorm, Ymean = normalizeRatings(Y, R)

num_movies, num_users = Y.shape
num_features = 100

# Set Initial Parameters (W, X) use tf.Variable to track these variables

tf.random.set_seed(1234)
W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float32), name='W' )
X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float32), name='X')
b = tf.Variable(tf.zeros((1, num_users), dtype=tf.float32), name='b')

# Instantiate an optimizer
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

"""
Training the Collaborative Filtering Model

Let's now train the collaborative filtering model. This will learn the parameters X, W, and b.

Custom Training Loop Requirement:
    The operations involved in learning w, b, and x simultaneously do not fall into the typical 'layers' 
    offered in the TensorFlow neural network package. Consequently, the flow used in Course 2: Model, 
    Compile(), Fit(), Predict(), are not directly applicable. Instead, we can use a custom training loop.

Gradient Descent Process:
    Recall from earlier labs the steps of gradient descent:
    
    Repeat until convergence:
        • Compute forward pass (calculate predictions and cost)
        • Compute the derivatives of the loss relative to parameters
        • Update the parameters using the learning rate and computed derivatives

TensorFlow Automatic Differentiation:
    TensorFlow has the marvelous capability of calculating the derivatives for you. This is shown below. 
    Within the tf.GradientTape() section, operations on TensorFlow Variables are tracked. When 
    tape.gradient() is later called, it will return the gradient of the loss relative to the tracked 
    variables. The gradients can then be applied to the parameters using an optimizer.

Benefits of tf.GradientTape():
    • Automatic computation of gradients for complex functions
    • Tracks operations on TensorFlow Variables automatically
    • Eliminates manual derivative calculations
    • Integrates seamlessly with TensorFlow optimizers
    • Essential for custom training loops in modern machine learning

Implementation Notes:
    This is a very brief introduction to a useful feature of TensorFlow and other machine learning 
    frameworks. Further information can be found by investigating "custom training loops" within 
    the framework of interest.
"""

iteration = 200

lambda_ = 1.5

for iter in range(iteration):
    with tf.GradientTape() as tape:
        cost_value = cofi_cost_func_vec(X, W, b, Ynorm, R, lambda_)

    # Compute gradients
    grads = tape.gradient(cost_value, [X, W, b])

    # Apply gradients using optimizer
    optimizer.apply_gradients(zip(grads, [X, W, b]))

    # Log periodic   cost
    if iter % 20 == 0:
        print(f"Iteration {iter} : Cost {cost_value}")

# Get the learned parameters X and W
X_trained = X.numpy()
W_trained = W.numpy()
b_trained = b.numpy()

# Make a prediction
p = np.matmul(X_trained, np.transpose(W_trained)) + b_trained

# Restore the mean
pm = p + Ymean

my_predictions = pm[:,0]

# sort predictions in descending order
ix = tf.argsort(my_predictions, direction='DESCENDING')

my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]

# Print the results
for i in range(17):
    j = ix[i]
    if j not in my_rated:
        print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movie_list[j]}')

print('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movie_list[i]}')