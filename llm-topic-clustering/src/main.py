# main.py

import os
import json
from preprocess import preprocess_texts
from embed import generate_embeddings
from cluster import cluster_documents, calculate_silhouette_score, visualize_clusters
from label import label_clusters

# Ensure the model and dataset are downloaded
os.system("python3 src/download_model.py")
os.system("python3 src/download_dataset.py")
 
# Load the dataset
with open('../data/sample_dataset.json', 'r') as f:
    dataset = json.load(f)
documents = dataset['documents']

# Preprocess the texts
preprocessed_texts = preprocess_texts(documents)

# Generate embeddings
embeddings = generate_embeddings(preprocessed_texts)

# Cluster the documents
clusters = cluster_documents(embeddings)

# Calculate and display the silhouette score
silhouette_score = calculate_silhouette_score(embeddings, clusters)
print(f"Silhouette Score: {silhouette_score}")

# Visualize the clusters
visualize_clusters(embeddings, clusters)

# Label the clusters
labels = label_clusters(clusters, preprocessed_texts)

# Print the results
for i, label in enumerate(labels):
    print(f"Cluster {i}: {label}")