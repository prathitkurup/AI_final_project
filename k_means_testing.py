# COPIED FROM https://www.geeksforgeeks.org/k-means-clustering-introduction/

# Example 1: K-Means Clustering by Hand
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def example1():
    X,y = make_blobs(n_samples = 500,n_features = 2,centers = 3,random_state = 23)

    fig = plt.figure(0)
    plt.grid(True)
    plt.scatter(X[:,0],X[:,1])
    plt.show()

    k = 3

    clusters = {}
    np.random.seed(23)

    for idx in range(k):
        center = 2*(2*np.random.random((X.shape[1],))-1)
        points = []
        cluster = {
            'center' : center,
            'points' : []
        }
        
        clusters[idx] = cluster
        
    print(clusters)

    plt.scatter(X[:,0],X[:,1])
    plt.grid(True)
    for i in clusters:
        center = clusters[i]['center']
        plt.scatter(center[0],center[1],marker = '*',c = 'red')
    plt.show()

    def distance(p1,p2):
        return np.sqrt(np.sum((p1-p2)**2))

    #Implementing E step 
    def assign_clusters(X, clusters):
        for idx in range(X.shape[0]):
            dist = []
            
            curr_x = X[idx]
            
            for i in range(k):
                dis = distance(curr_x,clusters[i]['center'])
                dist.append(dis)
            curr_cluster = np.argmin(dist)
            clusters[curr_cluster]['points'].append(curr_x)
        return clusters
            
    #Implementing the M-Step
    def update_clusters(X, clusters):
        for i in range(k):
            points = np.array(clusters[i]['points'])
            if points.shape[0] > 0:
                new_center = points.mean(axis =0)
                clusters[i]['center'] = new_center
                
                clusters[i]['points'] = []
        return clusters

    def pred_cluster(X, clusters):
        pred = []
        for i in range(X.shape[0]):
            dist = []
            for j in range(k):
                dist.append(distance(X[i],clusters[j]['center']))
            pred.append(np.argmin(dist))
        return pred

    clusters = assign_clusters(X,clusters)
    clusters = update_clusters(X,clusters)
    pred = pred_cluster(X,clusters)

    plt.scatter(X[:,0],X[:,1],c = pred)
    for i in clusters:
        center = clusters[i]['center']
        plt.scatter(center[0],center[1],marker = '^',c = 'red')
    plt.show()

# Example 2: K-Means Clustering with sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

def example2():
    X, y = load_iris(return_X_y=True)
    #Find optimum number of cluster
    sse = [] #SUM OF SQUARED ERROR
    for k in range(1,11):
        km = KMeans(n_clusters=k, random_state=2)
        km.fit(X)
        sse.append(km.inertia_)

    sns.set_style("whitegrid")
    g=sns.lineplot(x=range(1,11), y=sse)
    g.set(xlabel ="Number of cluster (k)", 
        ylabel = "Sum Squared Error", 
        title ='Elbow Method')
    plt.show()

    kmeans = KMeans(n_clusters = 3, random_state = 2)
    kmeans.fit(X)
    print(kmeans.cluster_centers_)
    pred = kmeans.fit_predict(X)
    print(pred)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(X[:,0],X[:,1],c = pred, cmap=cm.Accent)
    plt.grid(True)
    for center in kmeans.cluster_centers_:
        center = center[:2]
        plt.scatter(center[0],center[1],marker = '^',c = 'red')
    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")
        
    plt.subplot(1,2,2)   
    plt.scatter(X[:,2],X[:,3],c = pred, cmap=cm.Accent)
    plt.grid(True)
    for center in kmeans.cluster_centers_:
        center = center[2:4]
        plt.scatter(center[0],center[1],marker = '^',c = 'red')
    plt.xlabel("sepal length (cm)")
    plt.ylabel("sepal width (cm)")
    plt.show()

# example1()
example2()