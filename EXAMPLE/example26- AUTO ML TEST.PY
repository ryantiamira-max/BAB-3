# Example 3.26 - Modern AutoML Regressor
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Menggunakan dataset regresi standar pengganti Boston (load_diabetes)
X, y = load_diabetes(return_X_y=True)

# Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Inisialisasi dan jalankan AutoML Regressor
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Menampilkan tabel perbandingan skor & performa semua model regresi
print(models)
