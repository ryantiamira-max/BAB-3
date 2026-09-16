# Exercise 3.18: Fast AutoML Regression on California Housing Dataset
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 1. Load Dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# 2. Ambil sampel 2.000 baris data saja agar running instan (tidak hang/lama)
df_sample = df.sample(n=2000, random_state=42)
X = df_sample.drop(columns=['MedHouseVal'])
y = df_sample['MedHouseVal']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Jalankan LazyRegressor
print("Sedang memproses model (hanya butuh waktu beberapa detik)...")
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# 5. Tampilkan Hasil
print("\n--- Hasil Evaluasi AutoML ---")
print(models)