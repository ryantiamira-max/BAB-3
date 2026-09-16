# Exercise 3.1: Modify Example 3.1 to use six samples of X and y.
from sklearn import svm
X = [[170, 70, 10], [180, 80, 12], [170, 65, 8], [160, 55, 7],
     [175, 72, 9], [165, 58, 6]]  # 6 samples now: Height[cm], Weight[kg], Shoesize[UK]
y = [0, 0, 1, 1, 0, 1]    # Gender, 0: Male, 1: Female (6 labels)
clf = svm.SVC()
clf.fit(X, y)
p = clf.predict([[160, 60, 7]])
print(p)