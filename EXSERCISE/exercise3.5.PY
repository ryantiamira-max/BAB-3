# Exercise 3.5: Modify Example 3.7 so that it first trains the naive Bayes
# model, saves the model to a file, loads the model from the file, and
# makes a prediction with the model.
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import joblib

X, y = load_iris(return_X_y=True)
clf = GaussianNB()
clf.fit(X, y)

# Save the trained model to a file (serialization)
joblib.dump(clf, 'model.pkl')

# Load the model back from the file (deserialization)
clf2 = joblib.load('model.pkl')

# Make a prediction with the loaded model
p = clf2.predict([[5.0, 3.4, 1.5, 0.4]])
print(p)
