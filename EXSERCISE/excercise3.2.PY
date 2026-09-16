# Exercise 3.2: Modify Example 3.2 to use the third and fourth features
# (petal length and width) as X. Compare the results with Example 3.2.
from sklearn import svm, datasets
iris = datasets.load_iris()
# Take the third and fourth features: Petal length and Petal width
X = iris.data[:, 2:4]
y = iris.target
print(y)
clf = svm.SVC()
clf.fit(X, y)
# Predict the flower for a given Petal length and width
p = clf.predict([[5.4, 3.2]])
print(p)