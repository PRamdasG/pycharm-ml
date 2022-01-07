import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

X = np.array([[1],[2],[3]])
X_train = X
X_test = X

Y_train = np.array([3,2,4])
Y_test = np.array([3,2,4])

model = linear_model.LinearRegression()
model.fit(X_train,Y_train)

Y_predict = model.predict(X_test)

print("mean squared error: ",mean_squared_error(Y_test,Y_predict))
print("weights: ",model.coef_)
print("intercept: ",model.intercept_)

plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_predict)
plt.show()