from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['flower'] = iris.target
df.drop(['sepal length (cm)', 'sepal width (cm)', 'flower'],axis='columns',inplace=True)

km = KMeans(n_clusters=3)
y_pred = km.fit_predict(df)

df['cluster'] = y_pred

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color = 'green')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color = 'blue')
plt.scatter(df3['petal length (cm)'],df3['petal width (cm)'],color = 'yellow')
plt.show()

'''ELBOW PLOT'''
sse = []
for i in range(1,10):
    km = KMeans(n_clusters=i)
    km.fit(df)
    sse.append(km.inertia_)
plt.plot(range(1,10),sse)
plt.show()