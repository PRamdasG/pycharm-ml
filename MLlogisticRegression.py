import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/7_logistic_reg/insurance_data.csv')
print(df)
X_train,X_test,y_train,y_test = train_test_split(df[['age']],df.bought_insurance,test_size=0.1)
model = LogisticRegression()
model.fit(X_train,y_train)
print(X_test)
y_pred = model.predict(X_test)
print(y_pred)
print(model.predict_proba(X_test))
# plt.scatter(X_train,y_train)
# plt.plot(X_test,y_pred)
# plt.show()