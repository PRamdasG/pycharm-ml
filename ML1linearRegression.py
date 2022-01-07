import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model #datasets are imported to work on it
from sklearn.metrics import mean_squared_error
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:,np.newaxis,2]   #for one variable we are slicing it and then taking the second one
                                             #Simply put, numpy.newaxis is used to increase the dimension
                                             # of the existing array by one more dimension, when used once.

diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_Y_train = diabetes.target[:-30]           #target means it's the thing your trying to predict.
diabetes_Y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train,diabetes_Y_train)   #upto this the model is completed now we will test the model

diabetes_Y_predict = model.predict(diabetes_X_test)   #this is used to pridict the performance of model
print("The mean squared error: ",mean_squared_error(diabetes_Y_test,diabetes_Y_predict))
print("Weights: ",model.coef_)
print("Intercepts: ",model.intercept_)
print("model score",model.score(diabetes_X_test,diabetes_Y_test))

plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_predict)
plt.show()
# The mean squared error:  0.18010302594522304
# Weights:  [0.40414592]
# Intercepts:  -0.5944587489785538

#we are willing to push sometinh
