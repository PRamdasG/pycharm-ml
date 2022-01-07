import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/9_decision_tree/Exercise/titanic.csv')
newdf = df.drop(['PassengerId','Name','SibSp','Parch','Ticket', 'Cabin', 'Embarked'],axis='columns')
print(newdf.keys())
new_age = math.floor(newdf.Age.mean())
newdf.Age = newdf.Age.fillna(new_age)

le = LabelEncoder()
newdf.Sex = le.fit_transform(newdf.Sex)
x = newdf[['Pclass','Sex','Age','Fare']]
y = newdf.Survived

model = tree.DecisionTreeClassifier()

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
model.fit(X_train,y_train)
# print(X_train)
print(model.score(X_test,y_test))
print(model.predict(X_test))
print(model.predict([[2,1,45.0,56.344]]))