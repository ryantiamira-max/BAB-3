# Exercise 3.11: Linear Regression with More Data Points & Plot Decorators
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 1. Menambah lebih banyak data points (menggunakan NumPy agar lebih banyak & realistis)
np.random.seed(42)
x = np.linspace(0, 20, 50)  # 50 titik data dari 0 sampai 20
y = 1.2 * x + 3 + np.random.normal(0, 2, 50)  # Hubungan linear + sedikit noise

# 2. Perhitungan Linear Regression
slope, intercept, r, p, std_err = stats.linregress(x, y)
print("slope    :", slope)
print("intercept:", intercept)
print("r-squared:", r**2)

# 3. Fungsi Garis Regresi
def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))

# 4. Visualisasi Plot
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', alpha=0.7, label='Data points')
plt.plot(x, mymodel, color='red', linewidth=2, label=f'Best-fit line (y = {slope:.2f}x + {intercept:.2f})')

# 5. Menambahkan Elemen Sesuai Instruksi Soal
plt.xlabel('X (Feature)')
plt.ylabel('Y (Target)')
plt.title('Linear Regression with More Data Points')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Tampilkan grafik ke layar
plt.show()