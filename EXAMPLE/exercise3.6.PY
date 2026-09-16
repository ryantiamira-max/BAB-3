# Exercise 3.6: Modify Example 3.8 so that it uses 6 features and 2,000
# samples. Compare the results with Example 3.8.
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X, y = make_classification(n_samples=2000, n_features=6,
                           n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
print(X)
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
print(clf.predict([[0, 0, 0, 0, 0, 0]]))