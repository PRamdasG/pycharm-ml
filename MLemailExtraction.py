import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/14_naive_bayes/spam.csv')
# print(df.head())
# print(df.groupby('Category').describe())
df['spam'] = df['Category'].apply(lambda x:1 if x == 'spam' else 0 )
# print(df.head())
X_train, X_test, y_train, y_test = train_test_split(df.Message,df.spam)

v = CountVectorizer()
'''this CountVectorizer is used for converting the text into number vector'''
X_train_count = v.fit_transform(X_train.values)
# print(X_train_count)
'''in this we are converting the numbers into array matrics'''
# X_train_count.toarray()[:3]
'''MultinomialNB is used when we have discreat values'''

model = MultinomialNB()
model.fit(X_train_count,y_train)

emails = [
    'congratulations you have won 90% discount on biscuits',
    'Hey this is Zira your voice assistant'
]
'''wherever we are using v.transform, we are actually converting the email text into number vector'''

emails_counts = v.transform(emails)
print(model.predict(emails_counts))
X_test_count = v.transform(X_test)
print(model.score(X_test_count,y_test))

'''same thing can be done using the pipeline method'''

clf = Pipeline([('vec',CountVectorizer()),('nb',MultinomialNB())])
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
# print(clf.predict(X_test))