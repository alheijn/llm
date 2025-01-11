from sklearn.cluster import KMeans
import numpy as np

def cluster_documents(embeddings, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans.labels_, kmeans.cluster_centers_

def calculate_silhouette_score(embeddings, labels):
    from sklearn.metrics import silhouette_score
    return silhouette_score(embeddings, labels)