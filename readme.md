# Computational Linguistics - Project Objective
Use the Mistral-7B model to identify topics in a small dataset and cluster the documents
thematically. The focus is on implementing a simple practical solution optimized for the available
local resources on my M3 Macbook Air.
# Proposed Implementation Details
## Dataset:
- Choose a small dataset (e.g., 2,000–5,000 documents), e.g. a subset of 20 Newsgroups or
Reuters (or the provided combined news article dataset).
- Ensure the text length is reasonable (short documents or summaries) to reduce computational
load.
## Model:
- Use Mistral-7B-v0.3 from Hugging Face (recommended as a small LLM in slides).
## Pipeline:
- Text Preprocessing:
    - Use lightweight preprocessing to clean text (e.g., remove special characters, lowercasing).
    - Possible libraries: spaCy or re for minimal preprocessing.
- Embedding Generation:
    - Use Mistral-7B to generate embeddings for each document.
- Clustering:
    - Use a simple clustering method like k-means from scikit-learn on the generated embeddings.
- Topic Labeling:
    - Extract representative words/phrases from each cluster using TF-IDF or keyword extraction techniques.
## Technologies:
- Python
- Libraries:
    - Hugging Face Transformers for Mistral-7B.
    - scikit-learn for clustering.
## Adaptations for Local Execution
- Model Optimization
-   (Use a quantized version of Mistral-7B to significantly reduce memory requirements)
-   Run inference on CPU with batch processing to manage resource constraints.
- Dataset Size:
    - Process only a smaller batch of documents (e.g., 500) at a time to avoid memory overflow.
## Challenges & Solutions
- Model Size:
    - The Mistral-7B model itself - however, quantization and CPU optimization can make it feasible for local execution
- Clustering Interpretability:
    - Simplify clustering by using a smaller number of clusters (e.g., 5–10) and manually validate results.
- Storage (ensure enough local storage is available - should be the case)
## Eﬃciency Measurement
- Manual Cluster Inspection, reviewing clusters to confirm thematic consistency.