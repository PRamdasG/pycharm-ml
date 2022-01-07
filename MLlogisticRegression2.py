import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://raw.githubusercontent.com/codebasics/py/master/ML/7_logistic_reg/Exercise/HR_comma_sep.csv")

''' Now do some exploratory data analysis to figure out which variables have direct and clear impact
on employee retention (i.e. whether they leave the company or continue to work)'''
left = df[df.left==1]
print(left.shape)
retained = df[df.left==0]
print(retained.shape)
# print(df.groupby('left').mean())

''' Plot bar charts showing impact of employee salaries on retention'''
pd.crosstab(df.salary,df.left).plot(kind = 'bar')
'''Plot bar charts showing corelation between department and employee retention'''
pd.crosstab(df.Department,df.left).plot(kind = 'bar')

plt.legend(title='left')
# plt.show()

''' Now build logistic regression model using variables that were narrowed down in step 1'''
subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
# print(subdf.head())

salary_dummies = pd.get_dummies(subdf.salary,prefix='salary')
# print(salary_dummies)

df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')
# print(df_with_dummies)
df_with_dummies.drop('salary',axis='columns',inplace = True)
# print(df_with_dummies.head())

X = df_with_dummies
# print(X.head())
y = df.left
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8)

model = LogisticRegression()
model.fit(X_train,y_train)
print(model.predict(X_test))
print(model.score(X_test,y_test))