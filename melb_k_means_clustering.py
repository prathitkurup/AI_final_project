import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from branca.colormap import LinearColormap 
from sklearn.cluster import KMeans

def visualize_plot(df, k, centroids):
    # Visualize clusters using scatter plot
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap('viridis', k)
    plt.scatter(df['Longtitude'], 
                df['Lattitude'], 
                c=df['rank'], 
                cmap=cmap, 
                s=10, 
                alpha=0.6, 
                label='Cluster Points')
    plt.colorbar(ticks=centroids, label="Centroid Values")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f'K-Means Clustering of Melbourne House Prices (k={k})')
    plt.legend()
    plt.show()

def visualize_map(df, k, centroids):
    # Visualize clusters using a Folium map
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

    m.save('melbourne_map.html')

def run_k_means(df, features, k):
    X = df[features]
    
    # Step 1: Implement K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    df['cluster'] = kmeans.predict(X)

    # Step 2: Sort clusters by their centroid values
    centroids = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(-centroids)  
    
    # Step 3: Create a mapping from cluster index to rank
    cluster_rank = {cluster: rank for rank, cluster in enumerate(sorted_indices)}
    df['rank'] = df['cluster'].map(cluster_rank)  

    # Step 4: Assign colors based on rank
    viridis = LinearColormap(['#440154', '#3B528B', '#21908C', '#5DC863', '#FDE725'], vmin=0, vmax=k-1)
    df['color_map'] = df['rank'].map(lambda x: viridis(x))

    return centroids

if __name__ == "__main__":
    # Load the Melbourne housing dataset
    file_path = 'melb_data.csv'  # Replace with the correct file path
    df = pd.read_csv(file_path)

    # Step 1: Data Cleanup - Remove rows where Bedroom2 is NaN or 0
    df = df.dropna(subset=['Bedroom2'])
    df = df[df['Bedroom2'] > 0]

    # Step 2: Calculate price per square meter
    df['price_per_bedroom'] = df['Price'] / df['Bedroom2']
    features = ['price_per_bedroom']

    # Step 3: Drop rows with NaN or invalid price_per_sqm
    df = df.dropna(subset=features)

    # Step 4: Run K-means clustering
    centroids = run_k_means(df, features, 8)
    centroids = centroids.sort()

    # Step 5: Visualize the results
    visualize_plot(df, 8, centroids)
    # visualize_map(df, 8, centroids)
