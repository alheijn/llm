# main.py

import os
import random
import json
from pathlib import Path
from tqdm import tqdm
from preprocess import preprocess_texts
from embed import generate_embeddings
from cluster import cluster_documents, calculate_silhouette_score, visualize_clusters
from label import label_clusters

# set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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
with open('data/sample_dataset.json', 'r') as f:
    dataset = json.load(f)
documents = dataset['documents']

# Randomly select smaller amount of texts from the dataset 
# (since 2000 texts takes too long to process - each embedding takes about 2 minutes on my machine)
random.seed(42)  # For reproducibility
selected_documents = random.sample(documents, 10)


# Preprocess the texts
# preprocessed_texts = preprocess_texts(documents)
# print("Text preprocessing complete.")
print("Preprocessing texts...")
preprocessed_texts = [preprocess_texts([doc])[0] for doc in tqdm(selected_documents, desc="Preprocessing")]
print("Text preprocessing complete.")

# Generate embeddings
# embeddings = generate_embeddings(preprocessed_texts)
# print("Embedding generation complete.")
print("Generating embeddings...")
embeddings = [generate_embeddings([text])[0] for text in tqdm(preprocessed_texts, desc="Generating embeddings")]
print("Embedding generation complete.")

# Cluster the documents
print("Clustering documents...")
clusters = cluster_documents(embeddings)
print("Clustering complete.")

# Visualize the clusters
print("Visualizing clusters...")
visualize_clusters(embeddings, clusters)
print("Cluster visualization complete.")

# Label the clusters
print("Labeling clusters...")
labels = label_clusters(clusters, preprocessed_texts)
print("Cluster labeling complete.")

# Print the results
for i, label in enumerate(labels):
    print(f"Cluster {i}: {label}")