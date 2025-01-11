from sklearn.datasets import fetch_20newsgroups
import json

# Fetch the 20 Newsgroups dataset
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Use a subset of 2000 documents
documents = data['data'][:2000]

# Save the documents to a JSON file for later use
dataset = {"documents": documents}
with open('../data/sample_dataset.json', 'w') as f:
    json.dump(dataset, f)