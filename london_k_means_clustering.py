import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans

def run_k_means(df, features, k_max):
    X = df[features]
    # y = df['price_per_sqm']

    # Step 1: Find optimum number of clusters
    sse = [] # sum squared error
    for k in range(1,30):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        sse.append(km.inertia_)

    # Visualize elbow method
    sns.set_style("whitegrid")
    g=sns.lineplot(x=range(1,30), y=sse)
    g.set(xlabel ="Number of cluster (k)", 
        ylabel = "Sum Squared Error", 
        title ='Elbow Method')
    plt.show()

    # Based on the elbow method, we can see that the optimal number of clusters is 6


    # Step 2: Implementing K-Means Clustering
    # TODO: what is random state
    # kmeans = KMeans(n_clusters = 6, random_state = 42)
    # kmeans.fit(X)
    # print(kmeans.cluster_centers_)
    # pred = kmeans.fit_predict(X)
    # print(pred)

    # # plt.figure(figsize=(12,5))
    # # plt.subplot(1,2,1)
    # plt.scatter(X[:,0],X[:,1],c = pred, cmap=cm.Accent)
    # plt.grid(True)
    # for center in kmeans.cluster_centers_:
    #     center = center[:2]
    #     plt.scatter(center[0],center[1],marker = '^',c = 'red')
    # plt.xlabel("longitude")
    # plt.ylabel("latitude")
    # plt.show()



if __name__ == "__main__":
    df = pd.read_csv('cleaned_london_house_price_data.csv')
    print(df)
    df = df.iloc[:10000]
    df['price_per_sqm'] = df['history_price'] / df['floorAreaSqM']
    features = ['latitude', 'longitude', 'price_per_sqm']

    
    # Drop unnecessary columns here

    run_k_means(df, features, 15)