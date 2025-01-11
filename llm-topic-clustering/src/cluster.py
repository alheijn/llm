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
    clusters = kmeans.fit_predict(embeddings)
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
    from sklearn.metrics import silhouette_score
    return silhouette_score(embeddings, clusters)