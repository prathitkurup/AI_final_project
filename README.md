# README: Clustering of Home Prices in London and Melbourne

# Project Overview

The goal of this project is to analyze and cluster housing prices in London and Melbourne using machine learning techniques, specifically K-means clustering. Our primary objective is to identify clusters of homes with similar price ranges and visualize these clusters on maps of the respective cities. This project leverages unsupervised learning to explore patterns in the real estate market and understand how home prices are distributed across neighborhoods in these two cities.

# Project Objectives

Data Analysis and Preparation: Extract, clean, and preprocess housing data for London and Melbourne, focusing on key features such as property address, sale price, bedrooms, and square footage.

Clustering Using K-Means: Apply the K-means algorithm to identify clusters of homes with similar prices, treating each cluster as a potential "neighborhood" of similarly priced properties.

Visualization: Overlay the resulting clusters on real maps of London and Melbourne to visualize how well the clusters correspond to existing neighborhood boundaries.

Analysis and Insights: Compare and contrast the modeled clusters with actual neighborhoods to determine if the clustering method effectively reflects real world values.

# Project Dependencies

The following Python packages are required for the project:

pandas: For data manipulation and preprocessing.

numpy: For numerical computations.

scikit-learn: For applying the K-means clustering algorithm.

matplotlib: For visualizing plots and maps.

folium: For rendering interactive maps.

branca: For creating colormaps used in map visualizations.

# How to Run?

Run the london_k_means_clustering.py for the London cluster plot and map. 
Run the mlb_k_means_clustering.py for Melbourne cluster plot and map. 

You can alter k in both the scripts in the main function!

# Contact:

Shibali Mishra: smishra@bowdoin.edu
Prathit Kurup: pkrurup@bowdoin.edu
Stefan Langshur: slangshur@bowdoin.edu
