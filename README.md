# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Start
Steps

1. Initialize Centroids
2. Assign Points to Nearest Centroid
3. Update Centroids
4. Repeat and Return Labels and Centroids
5. Example Usage
   
Stop
## Program:
```py
'''
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Vikamuhan reddy.n 
RegisterNumber:  212223240181
'''
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/Shoba/Downloads/Mall_Customers.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:]) 
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])
y_pred = km.predict(data.iloc[:,3:])
data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label = "cluster0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="blue", label="cluster1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="green", label="cluster2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="purple", label="cluster3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="orange", label="cluster4")

plt.legend()
plt.title("Customer Segments")
plt.show()
```

## Output:
### data.head() ,data.info(),data.isnull().sum():
![image](https://github.com/vikamuhan-reddy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/144928933/250f25c9-61f8-4dbe-947b-a4eed865bd57)

### Elbow graph:
![image](https://github.com/vikamuhan-reddy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/144928933/8849887f-e7cf-42f2-a9ee-efb5003b1462)

### Kmeans clustering:
![image](https://github.com/vikamuhan-reddy/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/144928933/93492531-49f8-41b7-8dea-6f618627c94f)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
