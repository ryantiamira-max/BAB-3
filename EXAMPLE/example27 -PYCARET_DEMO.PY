# Example 3.27 The PyCaret_demo.py
import pandas as pd
from sklearn import datasets
from pycaret.classification import setup, compare_models

iris = datasets.load_iris(as_frame=True)
iris.data['Target'] = iris.target
df = iris.data

# Setup & Bandingkan Model
setup(data=df, target='Target', session_id=123, verbose=False)
best_model = compare_models()

print("\nModel Terbaik:")
print(best_model)