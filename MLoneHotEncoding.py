import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("https://raw.githubusercontent.com/codebasics/py/master/ML/5_one_hot_encoding/homeprices.csv")
print(df)

# using pandas for creating dummy variables

dummies = pd.get_dummies(df.town)
print('\n\n',dummies)

merge = pd.concat([df,dummies],axis='columns')
print(merge)
# # dummy variable trap resolution
final = merge.drop(['town','west windsor'],axis='columns')
print(final)
#
# #now for linear regression we need to create X and Y variables so
# #as price is our dependent variable we will store it in Y
#
X = final.drop(['price'],axis='columns')
print(X)
#
y = final.price
print(y)

model = linear_model.LinearRegression()
model.fit(X,y)
print(model.score(X,y))
#
print(model.predict([[3700,0,0]]))
#
# # using sklearn for one hot encoder
le = LabelEncoder()     #labelencoder converts text column into numbers
#
dfle = df
dfle.town = le.fit_transform(dfle.town)    #converted text into number
print(dfle)
#
X = dfle[['town','area']].values
Y = dfle.price.values
#
ct = ColumnTransformer([('town',OneHotEncoder(),[0])],remainder='passthrough')

X = ct.fit_transform(X)
X = X[:,1:]
print(X)
model.fit(X,Y)
print(model.predict([[1,0,3543]]))