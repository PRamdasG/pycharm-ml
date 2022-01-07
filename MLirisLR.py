from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

df = datasets.load_iris()   #SL,SW,PL,PW 1:STETOSA,2:VERSICOLOUR,3:VERGINICA
# print(df.DESCR)
# print(df)
model = linear_model.LogisticRegression()
X_train,X_test,y_train,y_test = train_test_split(df.data,df.target,test_size=0.2)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#printing confusion matrix
cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize = (10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')
plt.show()
print(model.score(X_test,y_test))