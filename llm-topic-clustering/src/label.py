from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_keywords_from_clusters(clusters, n_keywords=5):
    """
    Extract representative keywords from each cluster using TF-IDF.
    
    Args:
        clusters (dict): A dictionary where keys are cluster labels and values are lists of documents in that cluster.
        n_keywords (int): The number of keywords to extract from each cluster.
        
    Returns:
        dict: A dictionary with cluster labels as keys and lists of keywords as values.
    """
    cluster_keywords = {}
    
    for label, documents in clusters.items():
        # Create a TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Get the feature names (words)
        feature_names = vectorizer.get_feature_names_out()
        
        # Sum the TF-IDF scores for each word in the cluster
        summed_tfidf = np.sum(tfidf_matrix.toarray(), axis=0)
        
        # Get the indices of the top n_keywords
        top_indices = np.argsort(summed_tfidf)[-n_keywords:][::-1]
        
        # Extract the top keywords
        top_keywords = [feature_names[i] for i in top_indices]
        
        cluster_keywords[label] = top_keywords
    
    return cluster_keywords