import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target  #adding a column in df with the values of target
df['flower_name'] = df.target.apply(lambda x : iris.target_names[x])


#creating 3 dataframes for three different flowers
df0 = df[:50]
df1 = df[50:100]
df2 = df[100:150]

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='blue',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='red',marker='.')
plt.show()

plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='blue',marker='*')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='red',marker=',')
plt.show()

X = df.drop(['target', 'flower_name'],axis='columns')
y = df.target

model = SVC()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
print(model.predict(X_test))

#regularization(C)
model_C = SVC(C = 10)
model_C.fit(X_train,y_train)
print(model_C.score(X_test,y_test))

#gamma
model_G = SVC(gamma = 10)
model_G.fit(X_train,y_train)
print(model_G.score(X_test,y_test))

#kernel
model_linear_kernel = SVC(kernel = 'linear')
model_linear_kernel.fit(X_train,y_train)
print(model_linear_kernel.score(X_test,y_test))