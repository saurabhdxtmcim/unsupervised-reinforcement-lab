"""
K-means Clustering Algorithm Implementation

The K-means algorithm is a method to automatically cluster similar data points together.
Given a training set {x^(1), ..., x^(m)}, the goal is to group the data into K cohesive "clusters".

Algorithm Overview:
K-means is an iterative procedure that:
1. Starts by guessing the initial centroids
2. Refines this guess by:
   - Repeatedly assigning examples to their closest centroids
   - Recomputing the centroids based on the assignments

The algorithm consists of two main phases:
1. Cluster assignment step: Assign each data point to the closest centroid
   - idx[i] corresponds to the index of the centroid assigned to example i
   - idx = find_closest_centroids(X, centroids)

2. Move centroid step: Compute means based on centroid assignments
   - centroids = compute_centroids(X, idx, K)

The K-means algorithm will always converge to some final set of means for the centroids.
However, the converged solution may not always be ideal and depends on the initial 
setting of the centroids. In practice, the algorithm is usually run multiple times 
with different random initializations, and the solution with the lowest cost function 
value (distortion) is chosen.

Functions to implement:
- find_closest_centroid: Assigns each training example to its closest centroid
- compute_centroids: Computes the mean of each centroid using assigned points
"""
import numpy as np
import matplotlib.pyplot as plt


def find_closest_centroids(X, centroids):
    """
    Find the closest centroid for each training example in the cluster assignment phase.
    
    This function assigns every training example x^(i) to its closest centroid, given the current
    positions of centroids.
    
    Parameters:
    -----------
    X : ndarray, shape (m, n)
        Data matrix where each row is a training example
    centroids : ndarray, shape (K, n)  
        The locations of all centroids, where K is the number of centroids
        
    Returns:
    --------
    idx : ndarray, shape (m,)
        One-dimensional array that holds the index of the closest centroid 
        (a value in {0, ..., K-1}) for every training example.
        Note: Index range 0 to K-1 because Python list indices start at 0
        
    Algorithm:
    ----------
    For every example x^(i), we set:
        c^(i) := j  that minimizes  ||x^(i) - μ_j||^2
        
    Where:
    - c^(i) is the index of the centroid closest to x^(i) (corresponds to idx[i])
    - μ_j is the position (value) of the j'th centroid (stored in centroids)
    - ||x^(i) - μ_j|| is the L2-norm (Euclidean distance)
    """
    # Number of training examples
    m = X.shape[0]
    
    # Number of centroids
    K = centroids.shape[0]
    
    # Initialize array to hold centroid assignments
    idx = np.zeros(m, dtype=int)
    
    for i in range(m):

        # Array to hold distance between X[i] and each centroid [j]
        distance = []
        for j in range(K):
            # calculate the norm betweek Xi - centroid[j]
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm_ij)

        # Find the index of the closest centroid
        idx[i] = np.argmin(distance)

    return idx

# Test the function
X = np.array([[3, 3], [4, 3], [1, 1]])
centroids = np.array([[3, 2], [1, 1]])
idx = find_closest_centroids(X, centroids)
print(idx)


def compute_centroids(X, idx, K):
    """
    Compute the centroid means in the move centroid step of the K-means algorithm.
    
    Given assignments of every point to a centroid, this function recomputes, for each centroid,
    the mean of the points that were assigned to it.
    
    Parameters:
    -----------
    X : ndarray, shape (m, n)
        Data matrix where each row is a training example
    idx : ndarray, shape (m,)
        Array containing the index of the centroid assigned to each training example
    K : int
        Number of centroids
        
    Returns:
    --------
    centroids : ndarray, shape (K, n)
        New centroid locations computed as the mean of assigned points
        
    Algorithm:
    ----------
    For every centroid μ_k we set:
        μ_k = (1/|C_k|) * Σ x^(i) for i∈C_k
        
    Where:
    - C_k is the set of examples that are assigned to centroid k
    - |C_k| is the number of examples in the set C_k
    - The sum is over all training examples assigned to centroid k
    
    Example:
    --------
    If two examples x^(3) and x^(5) are assigned to centroid k=2, then:
        μ_2 = 1/2(x^(3) + x^(5))
    """
    # Get the number of features
    m, n = X.shape
    
    # Initialize centroids array
    centroids = np.zeros((K, n))
    
    for k in range(K):
        # Get the indices of the points assigned to centroid k
        indices = np.where(idx == k)[0]

        # Compute the mean of the points assigned to centroid k
        centroids[k] = np.mean(X[indices], axis=0)
    
    return centroids


def run_kMeans(X, initial_centroids, max_iters, plot_progress=False):
    """
    Run the K-means algorithm on a 2D dataset to demonstrate how K-means works.
    
    This function orchestrates the complete K-means clustering algorithm by calling
    the two main functions (find_closest_centroids and compute_centroids) in a loop
    until convergence or maximum iterations are reached.
    
    Parameters:
    -----------
    X : ndarray, shape (m, n)
        Data matrix where each row is a training example
    initial_centroids : ndarray, shape (K, n)
        Initial positions of the centroids
    max_iters : int
        Maximum number of iterations to run the algorithm
    plot_progress : bool, optional
        Whether to visualize the progress of the algorithm at each iteration
        
    Returns:
    --------
    centroids : ndarray, shape (K, n)
        Final centroid positions after convergence
    idx : ndarray, shape (m,)
        Final cluster assignments for each data point
        
    Algorithm Flow:
    ---------------
    1. Initialize centroids with the provided initial positions
    2. For each iteration:
       a. Cluster assignment step: Assign each data point to closest centroid
       b. Move centroid step: Recompute centroids as mean of assigned points
    3. Continue until max_iters or convergence
    
    Visualization:
    --------------
    When plot_progress=True, the function will produce a visualization that steps 
    through the progress of the algorithm at each iteration, showing:
    - Data points colored by cluster assignment
    - Current centroid positions (black X-marks)
    - Centroid movement path across iterations
    - Final converged centroids in the middle of colored clusters
    """
    # Get number of centroids
    K = initial_centroids.shape[0]
    
    # Initialize centroids
    centroids = initial_centroids.copy()
    
    # Initialize cluster assignments
    idx = np.zeros(X.shape[0])
    
    # Store centroid history for plotting movement
    centroid_history = [centroids.copy()]
    
    # Run K-means algorithm
    for i in range(max_iters):
        print(f"K-Means iteration {i+1}/{max_iters}")
        
        # Cluster assignment step: assign each data point to closest centroid
        idx = find_closest_centroids(X, centroids)
        
        # Move centroid step: compute new centroids
        centroids = compute_centroids(X, idx, K)
        
        # Store centroid positions for movement visualization
        centroid_history.append(centroids.copy())
        
        # Optional: plot progress
        if plot_progress:
            plot_kMeans_progress(X, centroids, idx, K, i, centroid_history)
    
    return centroids, idx


def plot_kMeans_progress(X, centroids, idx, K, iteration, centroid_history):
    """
    Plot the progress of K-means algorithm at each iteration with centroid movement trails.
    
    This function visualizes the current state of the K-means algorithm, showing:
    - Data points colored by their cluster assignments
    - Current centroid positions as black X-marks
    - Centroid movement trails showing how centroids moved
    - Iteration number in the title
    
    Parameters:
    -----------
    X : ndarray, shape (m, n)
        Data matrix where each row is a training example
    centroids : ndarray, shape (K, n)
        Current positions of the centroids
    idx : ndarray, shape (m,)
        Current cluster assignments for each data point
    K : int
        Number of centroids/clusters
    iteration : int
        Current iteration number
    centroid_history : list
        List of centroid positions from all previous iterations
    """
    # Create a new figure for each iteration
    plt.figure(figsize=(10, 8))
    
    # Define colors for different clusters
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple']
    
    # Plot data points colored by cluster assignment
    for k in range(K):
        # Get points assigned to cluster k
        cluster_points = X[idx == k]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[k % len(colors)], alpha=0.7, s=50, 
                       label=f'Cluster {k+1}', edgecolors='black', linewidth=0.5)
    
    # Plot centroid movement trails
    for k in range(K):
        # Extract path for centroid k
        path_x = [hist[k, 0] for hist in centroid_history]
        path_y = [hist[k, 1] for hist in centroid_history]
        
        # Plot the trail
        plt.plot(path_x, path_y, 'k-', alpha=0.5, linewidth=2)
        
        # Plot all previous centroid positions as small dots
        for i, hist in enumerate(centroid_history[:-1]):
            plt.scatter(hist[k, 0], hist[k, 1], 
                       marker='o', s=30, c='gray', alpha=0.6)
    
    # Plot current centroids as large black X-marks
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               marker='x', s=300, c='black', linewidths=4, 
               label='Current Centroids')
    
    # Plot initial centroids as red diamonds
    if len(centroid_history) > 0:
        plt.scatter(centroid_history[0][:, 0], centroid_history[0][:, 1], 
                   marker='D', s=100, c='red', 
                   label='Initial Centroids', edgecolors='black')
    
    # Set labels and title
    plt.xlabel('X1', fontsize=12)
    plt.ylabel('X2', fontsize=12)
    plt.title(f'K-Means Progress - Iteration {iteration + 1}\nCentroid Movement Visualization', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add text showing iteration
    plt.text(0.02, 0.98, f'Iteration: {iteration + 1}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Show the plot
    plt.show()


# Test k means algorithm with a better dataset
# Create a more interesting dataset with multiple clusters
np.random.seed(42)  # For reproducible results

# Generate sample data with 3 natural clusters
cluster1 = np.random.normal([2, 2], 0.5, (15, 2))
cluster2 = np.random.normal([6, 6], 0.5, (15, 2))  
cluster3 = np.random.normal([2, 6], 0.5, (15, 2))

# Combine all clusters
X = np.vstack([cluster1, cluster2, cluster3])

# Set initial centroids (not optimal to show movement)
initial_centroids = np.array([[1, 1], [3, 3], [5, 5]])

max_iters = 10
plot_progress = True

print("Running K-Means with better dataset...")
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress)
print(f"\nFinal centroids:\n{centroids}")
print(f"Final cluster assignments: {idx}")


