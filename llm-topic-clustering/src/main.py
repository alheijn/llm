# main.py

import pandas as pd
from preprocess import preprocess_text
from embed import generate_embeddings
from cluster import perform_clustering
from label import extract_labels

def main():
    # Load dataset
    data = pd.read_csv('data/sample_dataset.csv')
    
    # Preprocess text
    preprocessed_texts = preprocess_text(data['text'])
    
    # Generate embeddings
    embeddings = generate_embeddings(preprocessed_texts)
    
    # Perform clustering
    clusters = perform_clustering(embeddings, num_clusters=5)
    
    # Extract labels for each cluster
    labels = extract_labels(embeddings, clusters)
    
    # Display results
    for i, label in enumerate(labels):
        print(f"Cluster {i}: {label}")

if __name__ == "__main__":
    main()