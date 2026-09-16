# Exercise 3.10: Modify Example 3.14 so that it performs the comparison on
# Scikit-Learn's diabetes data.
# Note: the diabetes dataset target is continuous (a regression problem),
# so the classifiers in Example 3.14 are replaced with their regressor
# counterparts to keep a fair, direct comparison.
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

names = ["SVR", "Linear Regression", "Decision Tree",
         "Random Forest", "Nearest Neighbors", "Neural Networks"]

regressors = [
    SVR(),
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    KNeighborsRegressor(),
    MLPRegressor(alpha=1, max_iter=2000)]

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

for name, reg in zip(names, regressors):
    reg.fit(X_train, y_train)
    score = reg.score(X_test, y_test)
    print(name + ": " + str(score))
