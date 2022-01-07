import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB

df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/14_naive_bayes/titanic.csv')
# print(df.keys())
newdf = df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')
# print(newdf)
dummies = pd.get_dummies(newdf.Sex)
# print(dummies)
finaldf = pd.concat((newdf,dummies),axis='columns')
# print(finaldf)
finaldf.drop(['Sex'],axis='columns',inplace=True)
# print(finaldf)

finaldf.Age = finaldf.Age.fillna(finaldf.Age.mean())
# print(finaldf)
inputs = finaldf.drop(['Survived'],axis='columns')
# print(inputs)
target = finaldf.Survived
# print(target)
# print(inputs.isnull())
# print(target.isna())

X_train,X_test,y_train,y_test = train_test_split(inputs,target,test_size=0.2)

model = GaussianNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
print(model.predict(X_test))
# print(model.predict_proba(X_test))
model1 = MultinomialNB()
model1.fit(X_train,y_train)
print(model1.score(X_test,y_test))
print(model1.predict(X_test))
# print(model.predict_proba(X_test))