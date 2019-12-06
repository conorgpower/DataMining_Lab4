from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt

# Step 1
# Q. Import the Iris Dataset from SciKitLearn.
# A. 
def importData():
    iris = datasets.load_iris()
    return iris

# Step 2 
# Q. Use K-Means to build 2, 3, 4, … 10 clusters.
# A. 
def kMeansClusters(data):
    X = data.data
    Y = data.target
    kMeansArray = [[] for x in range(9)]

    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
        kMeansArray[i-2] = kmeans
        prediction = kmeans.predict(X)
        print(i, "Cluster KMeans Prediciton:\n", prediction)

    return kMeansArray

# Step 3
# Q. Plot values of the within cluster distance with respect to the number of clusters.
# A. 
def plotWithinClusterDistance(kMeansArray):
    within = []
    clusters = []

    for i in kMeansArray:
        within.append(i.inertia_)
        clusters.append(i.n_clusters)
    
    plt.scatter(clusters, within)
    plt.title("Within Cluster Distance")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Distance")
    plt.show()
        
# Step 4
# Q. Plot values of the between cluster distance with respect to the number of clusters.
# A. 
def plotBetweenClusterDistance(kMeansArray, X):
    center = KMeans(n_clusters=1, random_state=0).fit(X)

    within = []
    between = []

    for i in kMeansArray:
        values, counts = np.unique(i.labels_, return_counts=True)
        distance = 0
        for cluster, count in zip(i.cluster_centers_, counts):
            distance += np.dot(count, np.square(metrics.euclidean_distances([cluster], center.cluster_centers_)))
        
        between.append(distance)
        within.append(i.n_clusters)

    plt.scatter(within, between)
    plt.title("Between Clustering")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Distance")
    plt.show()

# Step 5
# Q. Plot values of the Calinski-Herbasz index with respect to the number of clusters.
# A. 
def plotCalinskiHerbasz(kMeansArray, X, title):
    calinskiHerbasz = []
    clusters = []

    for i in kMeansArray:
        calinskiHerbasz.append(metrics.calinski_harabasz_score(X, i.labels_))
        clusters.append(i.n_clusters)

    plt.scatter(clusters, calinskiHerbasz)
    plt.title(title)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Index")
    plt.show()

    return clusters, calinskiHerbasz

# Step 6
# Q. What is the natural cluster arrangement and why?
# A. 
def naturalClusterArrangementCH(clusters, calinskiHerbasz, name):
    naturalClusterArrangement(clusters, calinskiHerbasz, name)
# The natural cluster arrangement can be deduced from the results of the CH index.
# The peak value of the scatter plot corresponds with highest value in the Calinski-
# Herbasz array from Step 5. The CH Index states that this peak value is the value 
# of the natural cluster arrangement.

# Step 7
# Q. Use Hierarchical clustering to identify arrangement of the data-points.
# A. 
def hierarchicalClustering(X):
    hierarchicalCluster = []
    for i in range(2, 11):
        hierarchicalCluster.append(AgglomerativeClustering(n_clusters=i).fit(X))
    clusters, indexs = plotCalinskiHerbasz(hierarchicalCluster, X, "Hierarchical Cluster Index")
    return clusters, indexs

# Step 8 
# Q. What is the natural arrangement there and why?
# A. 
def naturalClusterArrangementHierarchical(clusters, indexs, name):
    naturalClusterArrangement(clusters, indexs, name)
# The natural cluster arrangement can be deduced from the results of the hierarchical
# clustering in step 7, in the same way as explained in step 6. The peak value of the 
# scatter chart from step 7 represents the natural cluster arrangement. 

# Method to find natural cluster arrangement from index values
def naturalClusterArrangement(clusters, values, name):
    peak = max(values)
    naturalClusterArrangement = clusters[values.index(peak)]
    print("The", name, "natural cluster arrangement is: ", naturalClusterArrangement)

def main():
    data = importData()
    kMeansArray = kMeansClusters(data)
    plotWithinClusterDistance(kMeansArray)
    plotBetweenClusterDistance(kMeansArray, data.data)
    clusters, calinskiHerbasz = plotCalinskiHerbasz(kMeansArray, data.data, "Calinski-Herbasz Index")
    naturalClusterArrangementCH(clusters, calinskiHerbasz, "Calinski-Herbasz")
    clusters, indexs = hierarchicalClustering(data.data)
    naturalClusterArrangementHierarchical(clusters, calinskiHerbasz, "Hierarchical")

main()