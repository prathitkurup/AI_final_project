import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from branca.colormap import LinearColormap
from sklearn.cluster import KMeans

def visualize_plot(df, k, centroids):
    """
    Visualize the K-means clustering results using a scatter plot.
    Colorbar is labeled with the centroid values.
    """
    if centroids is None:
        return
    
    plt.figure(figsize=(10, 8))
    
    # Use the viridis colormap
    cmap = plt.cm.viridis
    
    # Scatter plot with cluster ranks
    plt.scatter(
        df['Longtitude'],
        df['Lattitude'],
        c=df['rank'], 
        cmap=cmap, 
        s=10, 
        alpha=0.6, 
        label='Cluster Points'
    )
    
    # Custom colorbar with sorted centroid labels
    cbar = plt.colorbar()
    # Define the tick positions
    cbar.set_ticks(range(k))  
    # Set labels to centroid values
    cbar.set_ticklabels([f"{centroid:.2f}" for centroid in centroids])  
    cbar.set_label("Mean Price per Bedroom")

    plt.xlabel("Longtitude")
    plt.ylabel("Lattitude")
    plt.title(f'K-Means Clustering of House Prices in Melbourne (k={k})')
    plt.legend()
    plt.show()


def visualize_map(df, k, centroids):
    """
    Visualize the clustering results using an interactive folium map.
    """
    m = folium.Map(location=[-37.81, 144.96], zoom_start=10)
    
    for i in range(k):
        cluster_df = df.loc[df['cluster'] == i]
        for _, row in cluster_df.iterrows():
            folium.CircleMarker(
                location=[row['Lattitude'], row['Longtitude']],
                radius=5,
                color=row['color_map'],  
                fill=True,
                fill_color=row['color_map'],  
                fill_opacity=0.3,
                opacity=0.3
            ).add_to(m)

    m.save('melb_map.html')


def run_k_means(df, features, k):
    """
    Run K-means clustering on the selected features and rank clusters by their centroid values.
    """
    X = df[features]
    
    # Step 1: Implement K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    df['cluster'] = kmeans.predict(X)

    # Step 2: Calculate centroids and sort descending
    centroids = kmeans.cluster_centers_.flatten()
    sorted_centroids = np.sort(centroids)[::-1]  

    # Step 3: Map clusters to ranks
    sorted_indices = np.argsort(-centroids) 
    cluster_rank = {cluster: rank for rank, cluster in enumerate(sorted_indices)}
    df['rank'] = df['cluster'].map(cluster_rank)

    # Step 4: Assign colors to clusters
    viridis = LinearColormap(['#440154', '#3B528B', '#21908C', '#5DC863', '#FDE725'], vmin=0, vmax=k-1)
    df['color_map'] = df['rank'].map(lambda x: viridis(x))

    return sorted_centroids


if __name__ == "__main__":
    # Load the Melbourne housing dataset
    file_path = 'clean_melb_data.csv'  # Replace with the correct file path
    df = pd.read_csv(file_path)

    # Step 2: Calculate price per square meter
    df['price_per_bedroom'] = df['Price'] / df['Bedroom2']
    features = ['price_per_bedroom']

    # Run K-means clustering and visualize the results
    k = 8  
    centroids = run_k_means(df, features, k)
    visualize_plot(df, k, centroids)
    visualize_map(df, k, centroids)
    