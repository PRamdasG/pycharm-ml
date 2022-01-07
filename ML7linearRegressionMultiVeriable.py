import pandas as pd
from sklearn import linear_model
import sqlalchemy
import math
import pickle


#data preprocessing step:- cleaning the data
engine = sqlalchemy.create_engine('mysql://root:1234@localhost:3306/pradumnya')
df = pd.read_sql_table("houseprices",engine)
# print(df)
median_bedrooms = math.floor(df.bedroom.median())

# print(median_bedrooms )
df.bedroom = df.bedroom.fillna(median_bedrooms)
print(df)

#linear Regression
model = linear_model.LinearRegression()
model.fit(df[['area','bedroom','age']],df.prices)


print(model.coef_)
print(model.intercept_)
print(model.predict([[3000,3,40]]))
print(model.predict([[2500,4,5]]))

# pickling the model for future use
with open('model_pickle','wb') as f:
    pickle.dump(model,f)

with open('model_pickle','rb') as f:
    mp = pickle.load(f)

print(mp.predict([[3000,1,90]]))