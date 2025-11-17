import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.cleaning import load_and_clean_data
from src.features import select_and_normalize_features
from src.modeling import elbow_method, perform_clustering, evaluate_clusters, visualize_clusters, identify_anomalies

# ==========================
# Main Workflow
# ==========================

# 1. Load and Clean Data
print("1. Loading and cleaning data...")
df = load_and_clean_data("data/dataset.csv")
print("Data loaded and cleaned. Shape:", df.shape)

# 2. Select and Normalize Features
print("2. Selecting and normalizing features...")
scaled_features, scaler = select_and_normalize_features(df)
print("Features selected and normalized.")

# 3. Elbow Method
print("3. Performing Elbow Method...")
elbow_method(scaled_features)
print("Elbow method plotted. Check data/elbow_method.png")

# 4. K-Means Clustering
print("4. Performing K-Means Clustering...")
df = perform_clustering(scaled_features, df, k=3)
print("Clustering performed.")

# 5. Evaluate Clusters
print("5. Evaluating clusters...")
score = evaluate_clusters(scaled_features, df)
print("Clusters evaluated. Silhouette Score:", score)

# 6. Visualize Clusters
print("6. Visualizing clusters...")
visualize_clusters(df)
print("Clusters visualized. Check data/cluster_visualization.png")

# 7. Identify Anomalies
print("7. Identifying anomalies...")
anomali = identify_anomalies(df)
print("Anomalies identified. Number:", len(anomali))
