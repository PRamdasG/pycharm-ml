from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree

iris = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2)
score1 = cross_val_score(LogisticRegression(),iris.data,iris.target,cv=3)
score2 = cross_val_score(SVC(),iris.data,iris.target,cv=3)
score3 = cross_val_score(RandomForestClassifier(n_estimators=10),iris.data,iris.target,cv=3)
score4 = cross_val_score(tree.DecisionTreeClassifier(),iris.data,iris.target,cv=3)

print(score1)
print(score2)
print(score3)
print(score4)