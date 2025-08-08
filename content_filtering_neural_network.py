import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
from tensorflow.keras.layers import Lambda
from helpers.recsysNN_utils import *
pd.set_option('display.precision', 2)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
top10_df = pd.read_csv(os.path.join(data_dir, "content_top10_df.csv"))
bygenre_df = pd.read_csv(os.path.join(data_dir, "content_bygenre_df.csv"))

#print the top 10 movies
print(top10_df)
#print the bygenre_df
print(bygenre_df)

"""
Content-Based Filtering with Neural Networks

Comparison with Collaborative Filtering:
    - Collaborative Filtering: Uses only user-item ratings to generate user and movie vectors
    - Content-Based Filtering: Uses additional information about users and movies to improve predictions

Neural Network Architecture:
    The content-based approach uses two parallel neural networks:
    1. User Network: Takes user content (xu) → processes through hidden layers → outputs user vector (vu)
    2. Movie Network: Takes movie/item content (xm) → processes through hidden layers → outputs movie vector (vm)
    
    Prediction: The dot product vu · vm gives the predicted rating

Training Data Composition:
    Movie Content Features:
        - Year the movie was released
        - Movie genres (one-hot encoded vector, 14 genres total)
        - Average rating (engineered feature from user ratings)
    
    User Content Features:
        - Per-genre average rating computed per user
        - Additional metadata: user_id, rating_count, rating_average (for interpretation)
    
    Training Strategy:
        - Uses all user ratings in the dataset
        - Some ratings are repeated to boost training examples for underrepresented genres
        - Split into user array and movie/item array with same number of entries
"""

# Load Data, set configuration variables

item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

num_user_features = user_train.shape[1] - 3 # remove user id, rating count and rating during training
num_item_features = item_train.shape[1] - 1 # remove movie id at training

uvs = 3 # user genre vector start
ivs = 3 # item genere vector start
u_s = 3 # start of columns to use in training, user
i_s = 1 # start of columns to use in training, item

print(f"num_user_features: {num_user_features}" )
print(f"Number of Training vectors: {len(item_train)}")
pprint_train(user_train, user_features, uvs,  u_s, maxcount=5)
pprint_train(item_train, item_features, ivs, i_s, maxcount=5, user=False)
print(f"y_train[:5]: {y_train[:5]}")

# Scale the training data
item_train_unscaled = item_train
user_train_unscaled = user_train
y_train_unscaled = y_train

scalerItem = StandardScaler()
scalerItem.fit(item_train)
item_train = scalerItem.transform(item_train)

scalerUser = StandardScaler()
scalerUser.fit(user_train)
user_train = scalerUser.transform(user_train)

scalerTarget = MinMaxScaler(feature_range=(-1, 1))
scalerTarget.fit(y_train.reshape(-1, 1))
y_train = scalerTarget.transform(y_train.reshape(-1, 1))

print(np.allclose(item_train_unscaled, scalerItem.inverse_transform(item_train)))
print(np.allclose(user_train_unscaled, scalerUser.inverse_transform(user_train)))


# Split the data into training and test sets

item_train, item_test = train_test_split(item_train, test_size=0.80, shuffle=True, random_state=42)
user_train, user_test = train_test_split(user_train, test_size=0.80, shuffle=True, random_state=42)
y_train, y_test = train_test_split(y_train, test_size=0.80, shuffle=True, random_state=42)

print(f"movie/item training data shape: {item_train.shape}")
print(f"movie/item test data shape: {item_test.shape}")

pprint_train(user_train, user_features, uvs, u_s, maxcount=5)

"""
Exercise 1: Neural Network Construction

Neural Network Architecture Overview:
    We will construct a neural network with two parallel networks combined by a dot product.
    The networks process user content and movie content separately, then combine their outputs.
    
    Key Design Notes:
        - Two networks that are combined by dot product
        - In this example, both networks will be identical 
        - Networks don't need to be the same - if user content was substantially larger than
          movie content, you might increase complexity of user network relative to movie network
        - In this case, content is similar, so networks are the same

Exercise 1 Requirements - Keras Sequential Model:
    Create a sequential model with the following architecture:
        1. First layer: Dense layer with 256 units and ReLU activation
        2. Second layer: Dense layer with 128 units and ReLU activation  
        3. Third layer: Dense layer with num_outputs units and linear (no) activation

Implementation Notes:
    - The provided code will use Keras functional API instead of sequential model
    - Functional API allows for more flexibility in how components are interconnected
    - Both approaches (sequential and functional) can achieve the same neural network architecture
    - Functional API is better suited for complex architectures like this dual-network design
"""

num_outputs = 32
tf.random.set_seed(1234)
user_NN = tf.keras.Sequential([
    tf.keras.layers.Dense(units=256, activation="relu", input_shape=(num_user_features,)),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=num_outputs, activation="linear")
])

item_NN = tf.keras.Sequential([
    tf.keras.layers.Dense(units=256, activation="relu", input_shape=(num_item_features,)),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=num_outputs, activation="linear")
])

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features,), name="input_user")
vu = user_NN(input_user)
#vu = tf.linalg.l2_normalize(vu, axis=1)
vu = Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vu)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features,), name="input_item")
vm = item_NN(input_item)
#vm = tf.linalg.l2_normalize(vm, axis=1)
vm = Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vm)

# compute the dot product of the two vectors
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the input of the model
model = tf.keras.Model([input_user, input_item], output)

model.summary()

# compile the model
cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=cost_fn)

# train the model
#model.fit([user_train, item_train], y_train, epochs=10, batch_size=128)
model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=10, batch_size=128)

# evaluate the model
model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)

"""
Predictions

Predictions for a New User

Overview:
    This section demonstrates how to use the trained neural network model to generate movie 
    recommendations for new users. The model can make predictions even for users who weren't 
    in the original training dataset.

New User Prediction Process:
    1. Create user content vector representing the new user's preferences
    2. Use the trained model to generate predictions for all movies
    3. Sort predictions to find the highest-rated movie recommendations
    4. Filter out movies the user has already rated (if any)
    5. Present top recommendations to the user

User Content Requirements:
    - Per-genre average ratings that represent user's taste preferences
    - Same format as training data (matching feature dimensions)
    - Ratings must be between 0.5 and 5.0 (inclusive, in half-step increments)
    - Example: [4.5, 2.0, 5.0, 1.0, 3.5, ...] for each genre

Interactive Features:
    - Users can modify the example user content to match their own preferences
    - Experiment with different genre preferences to see how recommendations change
    - Real-time feedback on how user preferences affect movie suggestions

Technical Implementation:
    - Use the same preprocessing (scaling) as training data
    - Generate user vector tiles to match item vector dimensions
    - Apply trained model to predict ratings for all available movies
    - Use helper functions to format and display results
"""

new_user_id = 5000
new_rating_ave = 0.0
new_action = 0.0
new_adventure = 5.0
new_animation = 0.0
new_childrens = 0.0
new_comedy = 0.0
new_crime = 0.0
new_documentary = 0.0
new_drama = 0.0
new_fantasy = 5.0
new_horror = 0.0
new_mystery = 0.0
new_romance = 0.0
new_scifi = 0.0
new_thriller = 0.0
new_rating_count = 3

user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                      new_action, new_adventure, new_animation, new_childrens,
                      new_comedy, new_crime, new_documentary,
                      new_drama, new_fantasy, new_horror, new_mystery,
                      new_romance, new_scifi, new_thriller]])

# generate and replicate the user vector to match the number moviews in the dataset
user_vecs = gen_user_vecs(user_vec, len(item_vecs))

# scalle our user and item vectors
susers_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# make a prediction
y_p = model.predict([susers_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# unscale y predictions
y_pu = scalerTarget.inverse_transform(y_p)

# sort the results, highest prediction first
sorted_idx = np.argsort(-y_pu, axis=0).reshape(-1).tolist() # negate to get the largest rating first
sorted_ypu = y_pu[sorted_idx]
sorted_items = item_vecs[sorted_idx] # unsing unscaled vectors for display

print_pred_movies(sorted_ypu, sorted_items, movie_dict, maxcount = 10)

# Let's look at the predictions for "user 2", one of the users in the data set. We can compare the predicted ratings with the model's ratings.

uid = 2

# from a set of user vectors, This is the same veector, transformed and repeated
user_vecs, y_vecs =  get_user_vecs(uid, user_train_unscaled, item_vecs, user_to_genre)

# scale outr user and item vectors
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# make a prediction
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# unscale y predictions
y_pu = scalerTarget.inverse_transform(y_p)

# sort the results, highest prediction first
sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist() # negate to get the largest rating first
sorted_ypu = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]
sorted_user = user_vecs[sorted_index]
sorted_y = y_vecs[sorted_index]

#print sorted predictions for movies rated by the user
print_existing_user(sorted_ypu, sorted_y.reshape(-1,1), sorted_user, sorted_items, ivs, uvs, movie_dict, maxcount = 50)

"""
Finding Similar Items

Feature Vector Interpretation:
    The neural network above produces two feature vectors: a user feature vector (vu) and a movie 
    feature vector (vm). These are 32-entry vectors whose values are difficult to interpret. 
    However, similar items will have similar vectors.

Similarity-Based Recommendations:
    This information can be used to make recommendations. For example, if a user has rated 
    "Toy Story 3" highly, one could recommend similar movies by selecting movies with similar 
    movie feature vectors.

Mathematical Similarity Measure:
    A similarity measure is the squared distance between two vectors v_m^(k) and v_m^(l):
    
    ||v_m^(k) - v_m^(l)||² = Σ(l=1 to n) (v_ml^(k) - v_ml^(l))²
    
    Where:
        - v_m^(k) and v_m^(l) are movie feature vectors for movies k and l
        - n is the feature vector dimension (32 in our case)
        - Smaller distances indicate more similar movies

Implementation Strategy:
    1. Extract movie feature vectors from trained model
    2. Calculate squared distances between target movie and all other movies
    3. Sort by distance (smallest = most similar)
    4. Recommend movies with smallest distances to user's highly-rated movies
    
Use Cases:
    - "Users who liked this movie also liked..." recommendations
    - Finding movies similar to user's favorites
    - Content discovery based on movie characteristics rather than user behavior
"""

def sq_dist(a, b):
    return np.sum((a - b) ** 2)

a1 = np.array([1.0, 2.0, 3.0]); b1 = np.array([1.0, 2.0, 3.0])
a2 = np.array([1.1, 2.1, 3.1]); b2 = np.array([1.0, 2.0, 3.0])
a3 = np.array([0, 1, 0]);       b3 = np.array([1, 0, 0])
print(f"squared distance between a1 and b1: {sq_dist(a1, b1):0.3f}")
print(f"squared distance between a2 and b2: {sq_dist(a2, b2):0.3f}")
print(f"squared distance between a3 and b3: {sq_dist(a3, b3):0.3f}")

"""
Creating a Distance Matrix for Efficient Recommendations

Matrix-Based Approach:
    A matrix of distances between movies can be computed once when the model is trained and then 
    reused for new recommendations without retraining. This provides significant computational 
    efficiency for real-time recommendation systems.

Step 1: Extract Movie Feature Vectors
    The first step, once a model is trained, is to obtain the movie feature vector (vm) for each 
    of the movies. To do this, we will use the trained item_NN and build a small model to allow 
    us to run the movie vectors through it to generate vm.

Implementation Process:
    1. Create a separate model using only the trained item_NN network
    2. Pass all movie content vectors through this model to generate feature vectors
    3. Compute pairwise squared distances between all movie feature vectors
    4. Store distance matrix for fast lookup during recommendations

Benefits of Pre-computed Distance Matrix:
    - No need to retrain model for similarity-based recommendations
    - Fast lookup of similar movies for any given movie
    - Efficient scaling for large movie catalogs
    - Real-time recommendations without computational overhead

Use Case Example:
    When a user rates a movie highly, instantly recommend the movies with smallest 
    distances in the pre-computed matrix without any model inference.
"""

input_item_m = tf.keras.layers.Input(shape=(num_item_features,))  #  trailing comma makes it a tuple # input layer
vm_m = item_NN(input_item_m)                                       # use the trained item_NN
vm_m = Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vm_m)    # incorporate normalization as was done in the original model
model_m = tf.keras.Model(input_item_m, vm_m)                                
model_m.summary()

scaled_item_vecs = scalerItem.transform(item_vecs)
vms = model_m.predict(scaled_item_vecs[:,i_s:])
print(f"size of all predicted movie feature vectors: {vms.shape}")

count = 50  # number of movies to display
dim = len(vms)
dist = np.zeros((dim,dim))

for i in range(dim):
    for j in range(dim):
        dist[i,j] = sq_dist(vms[i, :], vms[j, :])
        
m_dist = ma.masked_array(dist, mask=np.identity(dist.shape[0]))  # mask the diagonal

disp = [["movie1", "genres", "movie2", "genres"]]
for i in range(count):
    min_idx = np.argmin(m_dist[i])
    movie1_id = int(item_vecs[i,0])
    movie2_id = int(item_vecs[min_idx,0])
    disp.append( [movie_dict[movie1_id]['title'], movie_dict[movie1_id]['genres'],
                  movie_dict[movie2_id]['title'], movie_dict[movie1_id]['genres']]
               )
#table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
table = tabulate.tabulate(disp, tablefmt="fancy_grid", headers="firstrow", floatfmt=[".1f", ".1f", ".0f", ".2f", ".1f"])

print(table)