# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

"""
Simple PCA Example - Easy to Understand and Visualize

This example demonstrates Principal Component Analysis (PCA) step by step
with clear visualizations and explanations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib

# Use non-interactive backend for better compatibility
matplotlib.use('Agg')

# Step 1: Create simple 2D data that has a clear pattern
print("Step 1: Creating sample data...")
np.random.seed(42)  # For reproducible results

# Create data that's spread along a diagonal (has correlation)
X = np.random.randn(100, 2)  # Start with random data
X[:, 1] = X[:, 0] + 0.5 * np.random.randn(100)  # Make y correlated with x
X[:, 0] = 2 * X[:, 0]  # Stretch x-axis

print(f"Original data shape: {X.shape}")
print(f"Data range - X: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
print(f"Data range - Y: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")

# Step 2: Visualize the original data
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.7, c='blue')
plt.title('Step 2: Original 2D Data')
plt.xlabel('Feature 1 (X)')
plt.ylabel('Feature 2 (Y)')
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Step 3: Apply PCA
print("\nStep 3: Applying PCA...")
pca = PCA(n_components=2)  # Keep both dimensions for now
X_pca = pca.fit_transform(X)

# Get the principal components (directions)
components = pca.components_
explained_variance_ratio = pca.explained_variance_ratio_

print(f"Explained variance ratio: {explained_variance_ratio}")
print(f"First PC explains {explained_variance_ratio[0]*100:.1f}% of variance")
print(f"Second PC explains {explained_variance_ratio[1]*100:.1f}% of variance")

# Step 4: Visualize original data with principal component directions
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], alpha=0.7, c='blue')
plt.title('Step 4: Data with Principal Components')
plt.xlabel('Feature 1 (X)')
plt.ylabel('Feature 2 (Y)')
plt.grid(True, alpha=0.3)

# Draw the principal component vectors
center = np.mean(X, axis=0)
for i, (component, variance) in enumerate(zip(components, explained_variance_ratio)):
    # Scale the arrow by the explained variance for better visualization
    scale = np.sqrt(variance) * 3
    plt.arrow(center[0], center[1], 
              component[0] * scale, component[1] * scale,
              head_width=0.3, head_length=0.2, 
              fc=f'C{i+1}', ec=f'C{i+1}', linewidth=2,
              label=f'PC{i+1} ({variance*100:.1f}%)')

plt.legend()
plt.axis('equal')

# Step 5: Show data in the new PCA coordinate system
plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c='green')
plt.title('Step 5: Data in PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.savefig('simple_pca_example.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved as 'simple_pca_example.png'")

# Step 6: Demonstrate dimensionality reduction
print("\nStep 6: Dimensionality Reduction...")
pca_1d = PCA(n_components=1)  # Reduce to 1 dimension
X_reduced = pca_1d.fit_transform(X)
X_reconstructed = pca_1d.inverse_transform(X_reduced)

print(f"Original data shape: {X.shape}")
print(f"Reduced data shape: {X_reduced.shape}")
print(f"Information retained: {pca_1d.explained_variance_ratio_[0]*100:.1f}%")

# Visualize the dimensionality reduction
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.7, c='blue', label='Original')
plt.title('Original 2D Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.subplot(1, 3, 2)
plt.scatter(range(len(X_reduced)), X_reduced[:, 0], alpha=0.7, c='red')
plt.title('Reduced to 1D')
plt.xlabel('Data Point Index')
plt.ylabel('Principal Component 1')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], alpha=0.4, c='blue', label='Original', s=20)
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.7, c='red', 
           label='Reconstructed', s=20)
plt.title('Original vs Reconstructed')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.savefig('pca_dimensionality_reduction.png', dpi=150, bbox_inches='tight')
print("Dimensionality reduction visualization saved as 'pca_dimensionality_reduction.png'")

# Summary
print("\n" + "="*60)
print("PCA SUMMARY:")
print("="*60)
print("1. PCA finds the directions (principal components) where data varies the most")
print("2. The first PC captures the most variance, second PC the second most, etc.")
print("3. PCA rotates your data to align with these principal components")
print("4. You can reduce dimensions by keeping only the most important components")
print(f"5. In this example, the first component captures {explained_variance_ratio[0]*100:.1f}% of the information")
print("6. This means we can represent 2D data with just 1D and lose very little information!")
print("="*60)