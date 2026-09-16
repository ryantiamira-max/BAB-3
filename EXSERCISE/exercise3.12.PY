# Exercise 3.12: Modify Example 3.18 so that it performs multiple linear
# regression on Scikit-Learn's Linnerud dataset.
from sklearn import linear_model
from sklearn.datasets import load_linnerud

linnerud = load_linnerud()
x = linnerud.data     # exercise data: Chins, Situps, Jumps
y = linnerud.target   # physiological data: Weight, Waist, Pulse

reg = linear_model.LinearRegression()
reg.fit(x, y)
print('Coefficients: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)
pred = reg.predict([[5, 8, 10]])
print('Predition: \n', pred)