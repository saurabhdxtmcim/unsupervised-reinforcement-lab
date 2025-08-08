# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from helpers.pca_utils import plot_widget
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
import matplotlib.pyplot as plt
import plotly.offline as py

X = np.array([[99, -1], [98, -1], [97, -2], [96, -3], [95, -2], [94, -3], [101, 1], [102, 1], [103, 2], [104, 1], [105, 2], [106, 3]])
plt.plot(X[:,0], X[:,1], 'o')
plt.show()

# Loading PCA algorithm
pca_2 = PCA(n_components=2)

# lets fit the data, We don not need to scale it, since sklearn's PCA does it internally.
pca_2.fit(X)
print(pca_2.explained_variance_ratio_)

# Lets transform the data
X_trans_2 = pca_2.transform(X)

# Lets plot the data
plt.plot(X_trans_2[:,0], X_trans_2[:,1], 'o')
plt.show()

"""
Interpretation of Principal Components:

- Column 1: Represents the coordinate along the first principal component (the first new axis).
- Column 2: Represents the coordinate along the second principal component (the second new axis).

Dimensionality Reduction Insight:
- The first principal component typically retains the majority of the dataset's variance (in this case, about 99%).
- Therefore, you can often use just the first principal component for further analysis or visualization, as it captures almost all the important information from the original data.
"""
pca_1 = PCA(n_components=1)
X_trans_1 = pca_1.fit_transform(X)
print(X_trans_1)

plt.plot(X_trans_1[:,0], X[:,1], 'o')
plt.show()

X_reduced_2 = pca_2.inverse_transform(X_trans_2)
print(X_reduced_2)

X_reduced_1 = pca_1.inverse_transform(X_trans_1)
print(X_reduced_1)

"""
Plots the reconstructed data from both 1D and 2D PCA in the same plot for comparison.

- X_reduced_1: Data reconstructed from 1 principal component.
- X_reduced_2: Data reconstructed from 2 principal components.
"""
plt.plot(X_reduced_1[:,0], X_reduced_1[:,1], 'o', label='1D PCA Reconstruction')
plt.plot(X_reduced_2[:,0], X_reduced_2[:,1], 'x', label='2D PCA Reconstruction')
plt.legend()
plt.show()

"""
Visualizing the PCA Algorithm

- We define 10 points in the 2D plane as an example dataset.
- The goal is to visualize how Principal Component Analysis (PCA) can compress these points from 2 dimensions down to 1 dimension.
- This process demonstrates dimensionality reduction: projecting data onto the direction (principal component) that retains the most variance (information).
- The plot will help illustrate both effective and ineffective ways to perform this compression.
"""

X = np.array([[-0.83934975, -0.21160323],
       [ 0.67508491,  0.25113527],
       [-0.05495253,  0.36339613],
       [-0.57524042,  0.24450324],
       [ 0.58468572,  0.95337657],
       [ 0.5663363 ,  0.07555096],
       [-0.50228538, -0.65749982],
       [-0.14075593,  0.02713815],
       [ 0.2587186 , -0.26890678],
       [ 0.02775847, -0.77709049]])

p = figure(title = '10-point scatterplot', x_axis_label = 'x-axis', y_axis_label = 'y-axis') ## Creates the figure object
p.scatter(X[:,0],X[:,1],marker = 'o', color = '#C00000', size = 5) ## Add the scatter plot

## Some visual adjustments
p.grid.visible = False
p.grid.visible = False
p.outline_line_color = None 
p.toolbar.logo = None
p.toolbar_location = None
p.xaxis.axis_line_color = "#f0f0f0"
p.xaxis.axis_line_width = 5
p.yaxis.axis_line_color = "#f0f0f0"
p.yaxis.axis_line_width = 5

## Shows the figure
show(p)

from helpers.pca_utils import random_point_circle, plot_3d_2d_graphs
X = random_point_circle(n = 150)
deb = plot_3d_2d_graphs(X)

deb.update_layout(yaxis2 = dict(title_text = 'test', visible=True))
deb.show()