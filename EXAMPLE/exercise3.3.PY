import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Load Dataset Iris
iris = load_iris(as_frame=True)
df = iris.frame

# 2. Buat Scatter Plot (Sepal length vs Sepal width)
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'])

# 3. Label dan Judul (Persis seperti di gambar)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Sepal length vs Sepal width')

# 4. Tampilkan Grafik
plt.show()