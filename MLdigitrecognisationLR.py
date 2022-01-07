import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn

ld = datasets.load_digits()
# for i in range(5):
#     plt.matshow(ld.images[i])
    # plt.show()
print(dir(ld))
model = LogisticRegression()
X_train,X_test,y_train,y_test = train_test_split(ld.data,ld.target,test_size=0.2)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
y_pred = model.predict(X_test)

#to print a confusion matrix
cm = confusion_matrix(y_test,y_pred)
# print(cm)

#using  seabron
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')
plt.show()