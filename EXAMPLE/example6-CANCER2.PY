# Example 3.6 Python SVM Breast Cancer Classifications
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

cancer = load_breast_cancer()
X = cancer.data  # All features
y = cancer.target  # All labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

clf = SVC()
clf.fit(X_train, y_train)

# Prediction
y_predict = clf.predict(X_test)

# Confusion Matrix & Classification Report
# Catatan: Di dataset ini, label 0 = Malignant (Cancer) dan label 1 = Benign (Healthy)
cm = np.array(confusion_matrix(y_test, y_predict, labels=[0, 1]))
confusion = pd.DataFrame(
    cm, 
    index=['is_cancer (0)', 'is_healthy (1)'],
    columns=['predicted_cancer', 'predicted_healthy']
)

print(confusion)
print("\nClassification Report:\n")
print(classification_report(y_test, y_predict))