# Exercise 3.9: Modify Example 3.12 so that it performs random forest
# classification on Scikit-Learn's diabetes data.
# Note: the diabetes dataset target is a continuous value (disease
# progression), so a RandomForestRegressor is used instead of a classifier.
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
reg = RandomForestRegressor()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("R-squared score on test data: %.4f" % reg.score(X_test, y_test))
