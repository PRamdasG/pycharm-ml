import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv("https://raw.githubusercontent.com/codebasics/py/master/ML/6_train_test_split/carprices.csv")
print(df)
X = df[['Mileage','Age(yrs)']]
y = df['Sell Price($)']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)

#lets run linear regression model now
model = LinearRegression()
model.fit(X_train,y_train)
model.predict(X_test)
print(X_test)
print(model.score(X_test,y_test))