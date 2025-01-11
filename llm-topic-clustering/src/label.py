from sklearn.feature_extraction.text import TfidfVectorizer

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
    vectorizer = TfidfVectorizer(max_features=10)
    labels = []
    for cluster in set(clusters):
        cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster]
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)
        terms = vectorizer.get_feature_names_out()
        labels.append(", ".join(terms))
    return labels
