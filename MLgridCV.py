from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd

iris = load_iris()

clf = GridSearchCV(SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel':['rbf','linear']
}, cv=5,return_train_score=False)
clf.fit(iris.data,iris.target)
# clf.cv_results_

df = pd.DataFrame(clf.cv_results_)
# print(df.keys())
print(df[['param_C', 'param_kernel','mean_test_score']])
