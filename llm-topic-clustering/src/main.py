# main.py

import os
import json
from pathlib import Path
from preprocess import preprocess_texts
from embed import generate_embeddings
from cluster import cluster_documents, calculate_silhouette_score, visualize_clusters
from label import label_clusters

# Ensure the spaCy model is downloaded
if not Path("llm-env/lib/python3.12/site-packages/en_core_web_sm").exists():
    os.system("python3 -m spacy download en_core_web_sm")

# Ensure the model is downloaded
model_path = Path("models/mistral-7b")
if not model_path.exists() or not any(model_path.iterdir()):
    os.system("python3 src/download_model.py")

# Ensure the dataset is downloaded
dataset_path = Path("data/sample_dataset.json")
if not dataset_path.exists():
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