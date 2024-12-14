import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import geopandas as gpd
from shapely.geometry import Point


def run_k_means(df, features, k):
    X = df[features]
    
    # Step 1: Implement K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    df['cluster'] = kmeans.predict(X)
    
    # Step 2: Assign colors to the clusters on a red-yellow spectrum
    cmap = plt.get_cmap('autumn', k)  # Get a colormap with k colors
    colors = [cmap(i) for i in range(k)]  # Extract k colors from the colormap
    df['color'] = df['cluster'].apply(lambda x: colors[x])
    
    # Step 3: Visualize clusters over map
    plt.figure(figsize=(10, 8))
    plt.scatter(df['longitude'], df['latitude'], c=df['color'], s=10, alpha=0.6, label='Cluster Points')
    plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=100, c='black', marker='^', label='Centers')
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
    df = df.iloc[:10000]
    
    # Calculate price per square meter and drop unnecessary columns
    df['price_per_sqm'] = df['history_price'] / df['floorAreaSqM']
    features = ['latitude', 'longitude', 'price_per_sqm']
    
    # Drop any rows with NaN values in the relevant columns
    df = df.dropna(subset=features)
    
    # Run K-means clustering and visualize the results on a map
    clustered_df = run_k_means(df, features, 8)
