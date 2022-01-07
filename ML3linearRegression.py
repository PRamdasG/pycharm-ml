import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
x = np.array([12,34,5,45,67,81]).reshape(-1,1)
y = np.array([21,45,51,55,37,91])

model = linear_model.LinearRegression()
model.fit(x,y)

y_pred = model.predict(x)

print("predicted response: ",y_pred)
print("weights: ",model.coef_)
print("intercept: ",model.intercept_)
print("mean squared error: ",mean_squared_error(y,y_pred))

plt.scatter(x,y)
plt.plot(x,y_pred)
plt.show()