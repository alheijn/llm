# from sklearn.cluster import KMeans
# import numpy as np

# def cluster_documents(embeddings, num_clusters=5):
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     kmeans.fit(embeddings)
#     return kmeans.labels_, kmeans.cluster_centers_

# def calculate_silhouette_score(embeddings, labels):
#     from sklearn.metrics import silhouette_score
#     return silhouette_score(embeddings, labels)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def cluster_documents(embeddings, n_clusters=10):
    """
    Cluster documents based on their embeddings using KMeans clustering.

    Args:
        embeddings (array-like): A 2D array where each row represents the embedding of a document.
        n_clusters (int, optional): The number of clusters to form. Default is 10.

    Returns:
        array: An array of cluster labels assigned to each document.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)   # labels
    return clusters

def calculate_silhouette_score(embeddings, clusters):
    """
    Calculate the silhouette score for the given embeddings and clusters.

    The silhouette score is a measure of how similar an object is to its own cluster 
    compared to other clusters. It ranges from -1 to 1, where a higher value indicates 
    that the object is well matched to its own cluster and poorly matched to neighboring clusters.

    Parameters:
    embeddings (array-like of shape (n_samples, n_features)): The feature embeddings of the samples.
    clusters (array-like of shape (n_samples,)): The cluster labels for each sample.

    Returns:
    float: The silhouette score.
    """
    return silhouette_score(embeddings, clusters)

def visualize_clusters(embeddings, clusters):
    """
    Visualize the clusters using PCA.

    Args:
        embeddings (array-like): A 2D array where each row represents the embedding of a document.
        clusters (array-like): An array of cluster labels assigned to each document.
    """
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title("Cluster Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()