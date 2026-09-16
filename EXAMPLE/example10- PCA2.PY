# Example 3.10 Principal Component Analysis Iris
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets

# Load Iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Plot Iris data (Scatter plot 2D dari fitur original)
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Original Data (2D View)')

# Perform PCA
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X1 = pca.transform(X)

# Plot PCA data (Scatter plot 2D dari 2 Komponen Utama pertama)
plt.figure(2)
plt.scatter(X1[:, 0], X1[:, 1], c=y, cmap='viridis')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA Data (2D View)')

# Menampilkan semua figure sekaligus
plt.show()