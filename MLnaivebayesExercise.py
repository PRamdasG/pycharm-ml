from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB

df = load_wine()
# print(df.data)
# print(df.feature_names)

X_train,X_test,y_train,y_test = train_test_split(df.data,df.target,test_size=0.2)
mnb = MultinomialNB()
gnb = GaussianNB()
bnb = BernoulliNB()

mnb.fit(X_train,y_train)
print("MultinomialNB()",mnb.score(X_test,y_test))
print(mnb.predict(X_test))

gnb.fit(X_train,y_train)
print("GaussianNB()",gnb.score(X_test,y_test))
print(gnb.predict(X_test))

bnb.fit(X_train,y_train)
print("BernoulliNB",bnb.score(X_test,y_test))
print(bnb.predict(X_test))