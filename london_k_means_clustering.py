from collections import Counter
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
        df['longitude'],
        df['latitude'],
        c=df['rank'], 
        cmap=cmap, 
        s=10, 
        alpha=0.6, 
        label='Cluster Points'
    )
    
    # Custom colorbar with sorted centroid labels
    cbar = plt.colorbar()
    cbar.ax.invert_yaxis()
    # Define the tick positions
    cbar.set_ticks(range(k))  
    # Set labels to centroid values
    cbar.set_ticklabels([f"{centroid:.2f}" for centroid in centroids])  
    cbar.set_label("Mean Price per Square Meter")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f'K-Means Clustering of House Prices in London (k={k})')
    plt.legend()
    plt.show()


def visualize_map(df, k, centroids):
    """
    Visualize the clustering results using an interactive folium map.
    """
    m = folium.Map(location=[51.5014, -0.140634], zoom_start=10)
    
    for i in range(k):
        cluster_df = df.loc[df['cluster'] == i]
        for _, row in cluster_df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=row['color_map'],  
                fill=True,
                fill_color=row['color_map'],  
                fill_opacity=0.3,
                opacity=0.3
            ).add_to(m)

    m.save('map.html')


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
    cluster_to_centroid = {i: centroid[0] for i, centroid in enumerate(kmeans.cluster_centers_)}
    sorted_centroids = np.sort(centroids)[::-1]

    # Step 3: Map clusters to ranks
    sorted_indices = np.argsort(-centroids) 
    cluster_rank = {cluster: rank for rank, cluster in enumerate(sorted_indices)}
    df['rank'] = df['cluster'].map(cluster_rank)

    # Step 4: Assign colors to clusters
    viridis = LinearColormap(['#440154', '#3B528B', '#21908C', '#5DC863', '#FDE725'], vmin=0, vmax=k-1)
    df['color_map'] = df['rank'].map(lambda x: viridis(x))

    return cluster_to_centroid, sorted_centroids

# Function to compute Euclidean distance
def euclidean_distance(lon1, lat1, lon2, lat2):
    # 1 degree of latitude is approximately 110 kilometers in London
    # 1 degree of longitude is approximately 70 kilometers in London
    return np.sqrt((70*(lon2 - lon1))**2 + (110*(lat2 - lat1))**2)

def predict_house_price(test_df, df, features, cluster_to_centroid):
    predictions = pd.DataFrame(columns=['longitude', 'latitude', 'floorAreaSqM', 'actual_price', 'centroid_val', 'predicted_price'])
    # Loop through the test data to predict the house price
    for i in range(len(test_df)):
        house = test_df.iloc[i]
        # Find the location of the house
        long = house['longitude']
        lat = house['latitude']

        # Apply the Euclidean distance function to each row in the DataFrame 
        df['distance'] = df.apply(
            lambda row: euclidean_distance(long, lat, row['longitude'], row['latitude']),
            axis=1
        )
        # Sort the DataFrame by distance and get the 10 nearest houses
        nearest_houses = df.nsmallest(10, 'distance')
        # Extract the labels of the 10 nearest houses
        nearest_labels = nearest_houses['cluster'].tolist()

        # Find the mode cluster label among the 10 nearest houses and get the centroid value
        cluster_label = Counter(nearest_labels).most_common(1)[0][0]
        centroid_val = cluster_to_centroid[cluster_label]

        # Calculate the predicted house price using the mean price per square meter of houses in the predicted cluster
        sq_m = house['floorAreaSqM']
        price = centroid_val * sq_m
        
        # Append the house details and the predicted price to the predictions DataFrame
        house_details = [long, lat, sq_m, house['history_price'], centroid_val, price]
        predictions.loc[len(predictions)] = house_details

    return predictions


if __name__ == "__main__":

    # Load the cleaned London house price data
    df = pd.read_csv('cleaned_london_house_price_data.csv')
    
    # Get a testing set for the house predictions
    test_df = df.iloc[:10]
    df = df.iloc[10:]
    
    # Calculate price per square meter and drop unnecessary columns
    df['price_per_sqm'] = df['history_price'] / df['floorAreaSqM']

    features = ['price_per_sqm']
    
    # Drop any rows with NaN values in the relevant columns
    df = df.dropna(subset=features)
    
    # Run K-means clustering and visualize the results on a map
    cluster_to_centroid, sorted_centroids = run_k_means(df, features, 8)

    visualize_map(df, 8, sorted_centroids)
    visualize_plot(df, 8, sorted_centroids)

    predictions = predict_house_price(test_df,df,features,cluster_to_centroid)
    print("Predictions: ")
    print(predictions)