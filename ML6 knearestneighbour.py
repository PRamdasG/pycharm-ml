#importing required modules
from sklearn import  datasets
from sklearn.neighbors import KNeighborsClassifier

#loading the datasets
iris = datasets.load_iris()
features = iris.data
labels = iris.target

#training the classifier
clf = KNeighborsClassifier()
clf.fit(features,labels)

#predicting the flower with the data
pred = clf.predict([[1.2,3.2,4.5,2.1]])
print(pred)
