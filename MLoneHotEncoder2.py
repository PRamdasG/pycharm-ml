import pandas as pd
from sklearn import linear_model

df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/5_one_hot_encoding/Exercise/carprices.csv')
print(df)
#creating dummy variable
dummies = pd.get_dummies(df['Car Model'] )
print(dummies)

merge = pd.concat([df,dummies],axis='columns')
# print(merge)

final = merge.drop(['Car Model','Mercedez Benz C class'],axis='columns')
# print(final)

X = final.drop(['Sell Price($)'],axis='columns')
# print(X)
y = df['Sell Price($)']
# print(y)

model = linear_model.LinearRegression()
model.fit(X,y)
print(model.score(X,y))
print(model.predict([[45000,4,0,0]]))
print(model.predict([[86000,7,0,1]]))