from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


digits = load_digits()
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.2)

#Cross Validation
print(cross_val_score(LogisticRegression(),digits.data,digits.target,cv=3))
print(cross_val_score(SVC(),digits.data,digits.target,cv=3))
print(cross_val_score(RandomForestClassifier(n_estimators=40),digits.data,digits.target,cv=3))
