import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from branca.colormap import LinearColormap 
from sklearn.cluster import KMeans

def visualize_plot(df, k):

    cmap = plt.get_cmap('viridis', k)  # Get a colormap with k colors
    colors = [cmap(i) for i in range(k)]  # Extract k colors from the colormap
    df['color'] = df['cluster'].apply(lambda x: colors[x])

    #for i in range(k):
    #    print(f"Cluster {i}: {df.loc[df['cluster'] == i].shape}")
    
    # Step 5: Visualize clusters over map
    plt.figure(figsize=(10, 8))
    plt.scatter(df['longitude'], df['latitude'], c=df['color'], s=10, alpha=0.6, label='Cluster Points')
    # plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=100, c='black', marker='^', label='Centers')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f'K-Means Clustering of House Prices in London (k={k})')
    plt.legend()
    # plt.show()

def visualize_map(df, k):
    # Step 4: Assign colors using Viridis colormap
    viridis = LinearColormap(['#440154', '#3B528B', '#21908C', '#5DC863', '#FDE725'], vmin=0, vmax=k-1)
    df['color'] = df['rank'].map(lambda x: viridis(x))


    # Step 5: Create a folium map
    m = folium.Map(location=[51.5014, -0.140634], zoom_start=10)
    for i in range(k):
        cluster_df = df.loc[df['cluster'] == i]
        for idx, row in cluster_df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=row['color'],  # Outline color
                fill=True,
                fill_color=row['color'],  # Fill color
                fill_opacity=0.3,  # Set fill opacity (30% transparent)
                opacity=0.3  # Set outline opacity (30% transparent)
            ).add_to(m)

    # Step 6: Save map (need to manually open)
    m.save('map.html')

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
    
    #Step 4: Visualize the Map
    visualize_map(df, k)
    visualize_plot(df, k)
    
    return df

if __name__ == "__main__":

    # Load the cleaned London house price data
    df = pd.read_csv('cleaned_london_house_price_data.csv')
    
    # Limit data for faster computation
    # df = df.iloc[:100]
    
    # Calculate price per square meter and drop unnecessary columns
    df['price_per_sqm'] = df['history_price'] / df['floorAreaSqM']
    features = ['price_per_sqm']
    
    # Drop any rows with NaN values in the relevant columns
    df = df.dropna(subset=features)
    
    # Run K-means clustering and visualize the results on a map
    clustered_df = run_k_means(df, features, 8)
