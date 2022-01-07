from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

digits = load_digits()
model = SVC()

X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.2)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
print(model.predict(X_test))

#regularization
model_r = SVC(C=10)
model_r.fit(X_train,y_train)
print(model_r.score(X_test,y_test))

#kernel
model_k = SVC(kernel='linear')
model_k.fit(X_train,y_train)
print(model_r.score(X_test,y_test))