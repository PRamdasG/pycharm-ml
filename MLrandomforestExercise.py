import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data)
df['target'] = iris.target

X = df.drop(['target'],axis='columns')
y = df.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier(n_estimators=30)

model.fit(X_train,y_train)
print(model.score(X_test,y_test))
y_pred = model.predict(X_test)
# print(model.predict([[12,34,45,23]]))


plt.figure(figsize=(10,7))
cm = confusion_matrix(y_test,y_pred)
sb.heatmap(cm,annot=True)
plt.ylabel('predicted')
plt.ylabel('truth')
plt.show()
