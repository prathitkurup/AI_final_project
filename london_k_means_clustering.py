import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import geopandas as gpd
from shapely.geometry import Point
import folium


def run_k_means(df, features, k):
    X = df[features]
    
    # Step 1: Implement K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    df['cluster'] = kmeans.predict(X)

    for i in range(k):
        print(f"Cluster {i}: {df.loc[df['cluster'] == i].shape}")

    # Step 2: Sort clusters by their centroid values
    centroids = kmeans.cluster_centers_.flatten()  
    sorted_indices = np.argsort(-centroids)  
    
    # Step 3: Create a mapping from cluster index to rank
    cluster_rank = {cluster: rank for rank, cluster in enumerate(sorted_indices)}
    df['rank'] = df['cluster'].map(cluster_rank)  # Assign rank to each cluster
    
    # Step 4: Assign colors based on rank (darker for higher centroids)
    cmap = plt.get_cmap('viridis', k)  
    colors = [cmap(i / (k - 1)) for i in range(k)]  
    df['color'] = df['rank'].map(lambda x: colors[x])
    
    # Step 5: Visualize clusters over map
    plt.figure(figsize=(10, 8))
    plt.scatter(df['longitude'], df['latitude'], c=df['color'], s=10, alpha=0.6, label='Cluster Points')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f'K-Means Clustering of House Prices in London (k={k})')
    plt.legend()
    plt.show()
    
    return df


if __name__ == "__main__":
    # Load the cleaned London house price data
    df = pd.read_csv('cleaned_london_house_price_data.csv')
    
    # Limit data for faster computation
    df = df.iloc[:1000]
    
    # Calculate price per square meter and drop unnecessary columns
    df['price_per_sqm'] = df['history_price'] / df['floorAreaSqM']
    features = ['price_per_sqm']
    
    # Drop any rows with NaN values in the relevant columns
    df = df.dropna(subset=features)
    
    # Run K-means clustering and visualize the results on a map
    clustered_df = run_k_means(df, features, 8)
