from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix

df = datasets.load_breast_cancer()
# print(df.data)
# print(df.target)
# print(df.DESCR)
X_train,X_test,y_train,y_test = train_test_split(df.data,df.target,test_size=0.2)

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(model.score(X_test,y_test))

cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')
plt.show()