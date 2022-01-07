import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/9_decision_tree/salaries.csv')
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

input = df.drop(['salary_more_then_100k'],axis='columns')
target = df['salary_more_then_100k']

input['le_company'] = le_company.fit_transform(input['company'])
input['le_job'] = le_job.fit_transform(input['job'])
input['le_degree'] = le_degree.fit_transform(input['degree'])

final = input.drop(['company','job','degree'],axis='columns')

#decision tree
model = tree.DecisionTreeClassifier()
clf = model.fit(final,target)
tree.plot_tree(clf)
# plt.legend(title = 'target')
# plt.show()

# print(model.score(final,target))
# print(model.predict([[2,1,1]]))