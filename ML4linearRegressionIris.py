import numpy as np
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

iris = datasets.load_iris()
iris_x = iris.data[:,np.newaxis,2]

iris_x_train = iris_x[:-70]
iris_x_test = iris_x[-70:]

iris_y_train = iris.target[:-70]
iris_y_test = iris.target[-70:]

model = linear_model.LinearRegression()
model.fit(iris_x_train,iris_y_train)
iris_y_pred = model.predict(iris_x_test)

print("Intercept: ",model.intercept_)
print("Weinghts: ",model.coef_)
print("mean squared error: ",mean_squared_error(iris_y_test,iris_y_pred))

plt.scatter(iris_x_test,iris_y_test)
plt.plot(iris_x_test,iris_y_pred)
plt.show()