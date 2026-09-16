# Exercise 3.19: Modify Example 3.27's PyCaret demo program to use the
# breast cancer dataset for classification.
import pandas as pd
from sklearn import datasets

cancer = datasets.load_breast_cancer(as_frame=True)
cancer.data['Target'] = cancer.target
cancer = cancer.data
cancer.head()

from pycaret import classification
classification.setup(data=cancer, target='Target')
classification.compare_models()
