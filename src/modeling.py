import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def elbow_method(scaled_features):
    """
    Perform Elbow Method to find optimal k.
    """
    wcss = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(range(2, 10), wcss, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Jumlah Cluster (k)")
    plt.ylabel("WCSS")
    plt.savefig('data/elbow_method.png')
    plt.close()

def perform_clustering(scaled_features, df, k=3):
    """
    Perform K-Means clustering and add cluster labels to df.
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled_features)
    return df

def evaluate_clusters(scaled_features, df):
    """
    Evaluate clusters using Silhouette Score.
    """
    score = silhouette_score(scaled_features, df["Cluster"])
    print("Silhouette Score:", score)
    return score

def visualize_clusters(df):
    """
    Visualize clusters with scatter plot.
    """
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        x=df["Quantity"],
        y=df["TotalPrice"],
        hue=df["Cluster"],
        palette="Set2"
    )
    plt.title("Cluster Hasil K-Means")
    plt.savefig('data/cluster_visualization.png')
    plt.close()

def identify_anomalies(df):
    """
    Identify anomalies as the smallest cluster.
    """
    cluster_sizes = df["Cluster"].value_counts()
    print("\nUkuran tiap cluster:\n", cluster_sizes)

    smallest_cluster = cluster_sizes.idxmin()
    anomali = df[df["Cluster"] == smallest_cluster]

    print("\nJumlah transaksi bermasalah:", len(anomali))
    print(anomali.head())
    return anomali
