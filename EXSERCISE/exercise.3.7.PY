# Exercise 3.7: PCA on Breast Cancer Dataset
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# 1. Load Data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# 2. Plot 1: Original Data (mean radius vs mean texture)
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('mean radius')
plt.ylabel('mean texture')
plt.title('Original Data')

# 3. Standarisasi Data sebelum PCA (Sangat Penting!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Pelaksanaan PCA
pca = decomposition.PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. Plot 2: PCA Data
plt.figure(2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA Data')

# Tampilkan kedua gambar ke layar
plt.show()
