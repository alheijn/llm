from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def label_clusters(clusters, texts):
    """
    Assigns labels to clusters based on the most significant terms in the texts.
    In other words: extracts representative keywords for each cluster.

    Parameters:
    clusters (list of int): A list of cluster assignments for each text.
    texts (list of str): A list of texts corresponding to the cluster assignments.

    Returns:
    list of str: A list of labels for each cluster, where each label is a comma-separated string of terms.
    """
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    labels = []
    unique_clusters = set(clusters)
    for cluster in tqdm(unique_clusters, desc="Labeling clusters"):
        # Extract the texts for the current cluster
        cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster]
        # Generate TF-IDF vectors for the texts
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)
        # Get the most significant terms
        terms = vectorizer.get_feature_names_out()
        # Append the terms as a comma-separated string to the labels list
        labels.append(", ".join(terms))

        # debug:
        print(f"Cluster {cluster}: {', '.join(terms)}")
    return labels