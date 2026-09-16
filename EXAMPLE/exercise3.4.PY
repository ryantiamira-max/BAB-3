# Exercise 3.4: Histograms of Breast Cancer Features
import pandas as pd
from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt

# 1. Load Dataset Breast Cancer
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# 2. Plot Histogram untuk 4 Fitur yang Diminta
# Tambahkan figsize agar ukuran jendela grafik proporsional & rapi
df[['mean radius', 'mean area', 'mean texture', 'mean smoothness']].hist(figsize=(10, 8), bins=20)

# 3. Setting Judul & Tata Letak
plt.suptitle('Histograms of Selected Breast Cancer Features', fontsize=14)
plt.tight_layout()  # Mencegah teks/label bertumpukan

# 4. Tampilkan Grafik ke Layar
plt.show()