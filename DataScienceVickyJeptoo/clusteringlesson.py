# modcom.co.ke/datascience
import pandas
df = pandas.read_csv('AirlinesCluster.csv')
print(df)
subset = df[['FlightMiles','FlightTrans','DaysSinceEnroll']]
array = subset.values
x = array[:,0:3]
# no target y, hence its unsupervised
from sklearn.cluster import KMeans
# elbow plot
# justpaste.it/26p9o
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
from sklearn.cluster import KMeans
wcss = [] # within cluster sum of squares
for i in range(1, 11):
     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42) # kmeans - object, KMeans - model
     kmeans.fit(x)
     wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss) # loop 10 times
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
# plt.show()

# from the elbow
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(x) # train unsupervised  way
# get the center number(mean of each cluster, for our 3 features
centronids =kmeans.cluster_centers_
dataframe = pandas.DataFrame(centronids, columns=['FlightMiles','FlightTrans','DaysSinceEnroll'])
print(dataframe)

# access the data for the clusters
result = zip(x, kmeans.labels_) # kmeans labels(group labels)
sorted_results = sorted(result, key = lambda x:x[1])
print(sorted_results)
for result in sorted_results:
    print(result)


