from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


digits = load_digits()
df = pd.DataFrame(digits.data)
# print(df)
df['target'] = digits.target
# print(df)

#training and testing the data
X = df.drop(['target'],axis='columns')
y = df.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model = RandomForestClassifier(n_estimators=1000)

model.fit(X_train,y_train)
print(model.score(X_test,y_test))
y_pred = model.predict(X_test)

plt.figure(figsize=(10,7))
cm = confusion_matrix(y_test,y_pred)
sb.heatmap(cm,annot=True)
plt.ylabel('predicted')
plt.ylabel('truth')
plt.show()
