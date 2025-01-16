from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class ClusterSummarizer:
    def __init__(self):
        # initialize a smaller, more efficient model for sentence embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_representative_texts(self, cluster_texts, cluster_labels, cluster_id, max_samples=10):
        """
        Select most representative texts using TF-IDF and keyword density
        
        Args:
            cluster_texts: list of texts in the current cluster
            cluster_labels: dict of cluster labels for all clusters
            cluster_id: identifier for the current cluster
            max_samples: maximum number of representative texts to return
        """        
        if not cluster_texts:
            return ["Empty cluster"]
        
        # Initialize TF-IDF vectorizer
        tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        
        # Get cluster TF-IDF matrix
        tfidf_matrix = tfidf.fit_transform(cluster_texts)
        
        # Calculate centroid
        centroid = tfidf_matrix.mean(axis=0).A1
        
        # Calculate similarity scores
        similarities = np.dot(tfidf_matrix.toarray(), centroid.T)
        
        # # Calculate keyword density scores
        # keywords = set(term.lower() for term in cluster_labels)
        # # AttributeError: 'int' object has no attribute 'lower'
        
        # Get keywords from cluster labels
        keywords = set()
        current_cluster_labels = cluster_labels.get(cluster_id, [])
        if isinstance(current_cluster_labels, list):
            for label in current_cluster_labels:
                if isinstance(label, str):
                    keywords.add(label)
        elif isinstance(current_cluster_labels, str):
            keywords.add(current_cluster_labels)
        
        # for term in cluster_labels.values():
        #     if isinstance(term, list):
        #         keywords.update(word.lower() for word in term)
        #     elif isinstance(term, str):
        #         keywords.add(term.lower())     
                   
        print(f"DEBUG cluster_id={cluster_id} keywords={keywords}")
        
        keyword_scores = []
        for text in cluster_texts:
            words = set(text.lower().split())
            density = len(words.intersection(keywords)) / len(words) if words else 0
            keyword_scores.append(density)
        
        # Combine scores
        final_scores = 0.7 * similarities.flatten() + 0.3 * np.array(keyword_scores)
        
        # Get indices of top scoring texts
        top_indices = final_scores.argsort()[-max_samples:][::-1]
        #print(f"DEBUG top_indices={top_indices}")
        
        #print(f"DEBUG cluster_texts[:4]={cluster_texts[:4]}")
        
        return [cluster_texts[i] for i in top_indices]
        