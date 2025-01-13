import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import numpy as np
import time
import psutil
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
import seaborn as sns
import platform
from sklearn.datasets import fetch_20newsgroups
import os
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import nltk
from datetime import datetime
import json


class DocumentClusterer:
    def __init__(self, model_id="mistralai/Mistral-7B-v0.3", num_clusters=5, batch_size=5):

        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.model_id = model_id
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        # Check if running on Apple Silicon
        self.is_apple_silicon = platform.processor() == 'arm'
        # Use MPS if available on Apple Silicon, otherwise CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.is_apple_silicon else "cpu")
        self.setup_model()
        # Initialize stemmer
        # stemmer is used to reduce words to their root form
        self.stemmer = PorterStemmer()

        # Download required NLTK (Natural Language Toolkit) data
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')

        # create output directories to store the results
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        for subdir in ['texts', 'plots', 'results']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        

    def save_texts(self, texts, categories=None):
        """Save input texts and their categories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, 'texts', f'input_texts_{timestamp}.json')
        
        data = {
            'texts': texts,
            'categories': categories if categories else []
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def save_clustering_results(self, clusters, cluster_labels, metrics, texts):
        """Save clustering results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, 'results', f'clustering_results_{timestamp}.json')
        
        results = {
            'num_clusters': self.num_clusters,
            'silhouette_score': metrics.get('silhouette_score', None),
            'runtime_seconds': metrics.get('runtime_seconds', None),
            'memory_usage_mb': metrics.get('memory_usage_mb', None),
            'cluster_labels': cluster_labels,
            'cluster_assignments': {
                i: {
                    'cluster': int(clusters[i]),
                    'text': texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i]
                }
                for i in range(len(texts))
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        
    def setup_model(self):
        """Initialize the model optimized for Apple Silicon"""
        print(f"Loading model and tokenizer (using {self.device} device)...")
        
        try:
            # Set memory management for MPS device
            if self.device.type == "mps":
                # Set memory fraction to 0.0 to enable dynamic memory allocation
                torch.mps.set_per_process_memory_fraction(0.0)
                print("Configured MPS memory management")
                
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})        
            
            # Load model with float16 on MPS for better memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                low_cpu_mem_usage=True
            )#.resize_token_embeddings(len(self.tokenizer))
            
            # Move model to appropriate device
            self.model.to(self.device)
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"\nError loading model: {str(e)}")
            print("\nTroubleshooting steps:")
            print("1. Ensure all dependencies are installed:")
            print("   pip install -r requirements.txt")
            print("2. Check if model files are correctly placed in models/mistral-7b/")
            print("3. Ensure you have enough system memory")
            print("4. Try reducing batch_size if you're running out of memory\n")
            raise
        
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        ### OLD CODE ###
        # # Convert to lowercase and remove special characters
        # text = text.lower()
        # text = re.sub(r'[^\w\s]', '', text)
        # tokens = text.split()
        # tokens = [self.stemmer.stem(token) for token in tokens if token not in ENGLISH_STOP_WORDS]
        # # Remove extra whitespace
        # text = ' '.join(tokens)
        # return text

        # convert to lowercase
        text = text.lower()
        # tokenize
        tokens = word_tokenize(text)
        # remove punctuations and numbers
        tokens = [token for token in tokens if token not in string.punctuation and not token.isnumeric()]
        # Remove stop words and short words
        tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in ENGLISH_STOP_WORDS and len(token) > 2
        ]
        # rejoin tokens
        return ' '.join(tokens)
        
    def get_embeddings(self, texts, show_progress=True):
        """Generate embeddings for texts in batches"""
        embeddings = []
        iterator = tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings: ") if show_progress else range(0, len(texts), self.batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i + self.batch_size]
            # Tokenize with padding
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use the mean of the last hidden state as the embedding
                batch_embeddings = outputs.hidden_states[-1].mean(dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())        

            # Clear GPU/MPS memory after each batch
            if self.device == "mps":
                torch.mps.empty_cache()
        
        return np.vstack(embeddings)
        # return embeddings
        
    def cluster_documents(self, embeddings):
        """Perform K-means clustering"""
        print(f"Clustering {len(embeddings)} documents into {self.num_clusters} clusters...")
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(embeddings, clusters)
        self.silhouette_avg = silhouette_avg
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        return clusters, kmeans
        
    def get_cluster_labels(self, texts, clusters, kmeans):
        """Extract representative terms for each cluster using TF-IDF"""
        tfidf = TfidfVectorizer(
            max_features=2000,
            stop_words=list(ENGLISH_STOP_WORDS),
            ngram_range=(1, 3),     # Include both single words and bigrams
            min_df=1,       # Term must appear in at least 2 documents
            max_df=0.90     # Ignore terms that appear in more than 90% of documents
        )

        # # Preprocess texts again before TF-IDF
        # processed_texts = [self.preprocess_text(text) for text in texts]
        # tfidf_matrix = tfidf.fit_transform(processed_texts)
        tfidf_matrix = tfidf.fit_transform(texts)
        
        cluster_labels = {}
        feature_names = tfidf.get_feature_names_out()

        for i in range(self.num_clusters):
            # Get texts in this cluster
            cluster_docs = tfidf_matrix[clusters == i]
            if cluster_docs.shape[0] == 0:
                cluster_labels[i] = ["Empty cluster"]   # new
                continue
                
            ### OLD CODE ###
            # # Get top terms
            # avg_tfidf = cluster_docs.mean(axis=0).A1
            # top_indices = avg_tfidf.argsort()[-5:][::-1]
            # top_terms = [tfidf.get_feature_names_out()[idx] for idx in top_indices]
            # cluster_labels[i] = top_terms

            # Calculate mean TF-IDF scores for the cluster
            avg_tfidf = cluster_docs.mean(axis=0).A1
            
            # Get candidate terms
            top_indices = avg_tfidf.argsort()[-30:][::-1]  # Get more terms initially
            candidates = [(feature_names[idx], avg_tfidf[idx]) for idx in top_indices]  # store term and its tf-idf score
            ## top_terms = []

            # Filter and group terms
            unigrams = []
            phrases = []
            
            # # Filter terms to ensure quality
            # for idx in top_indices:
            #     term = feature_names[idx]
            #     # Add term if it's not too short and has a significant TF-IDF score
            #     if (len(term) > 1 and  # Longer than 1 character
            #         avg_tfidf[idx] > 0.05 and  # Significant TF-IDF score
            #         not term.isnumeric()):  # Not just a number
            #         top_terms.append(term)
            #     if len(top_terms) >= 5:  # Keep top 5 meaningful terms
            #         break

            for term, score in candidates:
            # Separate unigrams and phrases
                if ' ' in term and score > 0.1:  # Higher threshold for phrases
                    phrases.append(term)
                elif score > 0.05:  # Lower threshold for single words
                    unigrams.append(term)
            
                # Stop if we have enough terms
                if len(phrases) >= 2 and len(unigrams) >= 3:
                    break
        
            # Combine phrases and unigrams for final labels
            top_terms = phrases[:2] + unigrams[:3]
            
            # if top_terms:  # Only add if we found meaningful terms
            #     cluster_labels[i] = top_terms
        
            # # Ensure cluster has labels
            # if not top_terms:
            #     # If no terms pass filters, use top 5 terms without filtering
            #     top_terms = [feature_names[idx] for idx in top_indices[:5]]
        
            # Ensure we have at least some terms
            if not top_terms and candidates:
                top_terms = [term for term, _ in candidates[:5]]
        
            cluster_labels[i] = top_terms                   
            
        return cluster_labels
    
    def visualize_clusters(self, embeddings, clusters, cluster_labels):
        """
        Create various visualizations for the clustering results.
        
        Args:
            embeddings (array): Document embeddings
            clusters (array): Cluster assignments
            cluster_labels (dict): Dictionary of cluster labels
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a figure with subplots
        fig = plt.figure(figsize=(20, 10))
        
        # 1. PCA visualization of clusters
        ax1 = fig.add_subplot(121)
        self._plot_cluster_pca(embeddings, clusters, cluster_labels, ax1)
        
        # 2. Cluster size distribution
        ax2 = fig.add_subplot(122)
        self._plot_cluster_sizes(clusters, cluster_labels, ax2)
        
        plt.tight_layout()
        # save combined plot
        plt.savefig(os.path.join(self.output_dir, 'plots', f'clusters_overview_{timestamp}.png'))
        plt.show()
        
        # 3. Term importance heatmap (separate figure)
        self._plot_term_heatmap(clusters, cluster_labels, timestamp)
        
    def _plot_cluster_pca(self, embeddings, clusters, cluster_labels, ax):
        """Plot PCA projection of document embeddings"""
        # Reduce dimensionality to 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Create scatter plot
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=clusters, cmap='viridis', alpha=0.6)
        
        # Add cluster centers
        for i in range(self.num_clusters):
            mask = clusters == i
            if np.any(mask):
                center = embeddings_2d[mask].mean(axis=0)
                if i in cluster_labels:
                    ax.annotate(f"Cluster {i}\n({', '.join(cluster_labels[i][:2])})",
                            xy=center, xytext=(10, 10),
                            textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->'))
        
        ax.set_title('Document Clusters (PCA Projection)')
        ax.set_xlabel(f'PC1 (Variance: {pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 (Variance: {pca.explained_variance_ratio_[1]:.2%})')
        
    def _plot_cluster_sizes(self, clusters, cluster_labels, ax):
        """Plot distribution of cluster sizes"""
        # Count documents per cluster
        cluster_sizes = Counter(clusters)
        
        # Create bars
        clusters_idx = list(cluster_sizes.keys())
        sizes = list(cluster_sizes.values())
        
        # Plot horizontal bars
        y_pos = np.arange(len(clusters_idx))
        bars = ax.barh(y_pos, sizes)
        
        # Customize plot
        ax.set_yticks(y_pos)
        labels = [f"Cluster {i}\n({', '.join(cluster_labels[i][:2])})" 
                 for i in clusters_idx]
        ax.set_yticklabels(labels)
        
        # Add value labels on bars
        for i, v in enumerate(sizes):
            ax.text(v + 1, i, str(v), va='center')
        
        ax.set_title('Cluster Size Distribution')
        ax.set_xlabel('Number of Documents')
        
    def _plot_term_heatmap(self, clusters, cluster_labels, timestamp):
        """Create heatmap of term importance across clusters"""
        # Create term importance matrix
        terms = set()
        for terms_list in cluster_labels.values():
            terms.update(terms_list)
        terms = sorted(terms)
        
        # Create matrix of term presence (1) or absence (0)
        matrix = np.zeros((len(cluster_labels), len(terms)))
        for i, cluster_terms in cluster_labels.items():
            for term in cluster_terms:
                matrix[i, terms.index(term)] = 1
                
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix, 
                   xticklabels=terms,
                   yticklabels=[f'Cluster {i}' for i in range(self.num_clusters)],
                   cmap='YlOrRd')
        plt.title('Term Importance Across Clusters')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', f'term_heatmap_{timestamp}.png'))
        plt.show()

        
    def process_dataset(self, dataset_name="20newsgroups", num_samples=2000):
        """Main processing pipeline"""
        # Load and preprocess dataset
        if dataset_name == "20newsgroups":
            dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=None)
            # Sample if needed
            if num_samples and num_samples < len(dataset.data):
                import random
                random.seed(42)
                indices = random.sample(range(len(dataset.data)), num_samples)
                texts = [dataset.data[i] for i in indices]
                categories = [dataset.target_names[dataset.target[i]] for i in indices]
                print(f"\nSampled categories: {set(categories)}")                
            else:
                texts = dataset.data
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        # Save input texts
        self.save_texts(texts, categories=categories if dataset_name == "20newsgroups" else None)

        # Track runtime and memory
        start_time = time.time()

        # Preprocess all texts
        texts = [self.preprocess_text(text) for text in texts]

        # Generate embeddings
        embeddings = self.get_embeddings(texts)
        
        # Perform clustering
        clusters, kmeans = self.cluster_documents(embeddings)
        
        # Get cluster labels
        cluster_labels = self.get_cluster_labels(texts, clusters, kmeans)
        print("\nCluster Topics:")
        for cluster_id, terms in cluster_labels.items():
            print(f"Cluster {cluster_id}: {', '.join(terms)}")
        
        # Calculate metrics
        end_time = time.time()
        final_memory = psutil.Process().memory_info().vms / 1024 / 1024  # MB       old: rss
        
        metrics = {
            "runtime_seconds": end_time - start_time,
            "memory_usage_mb": final_memory - self.initial_memory,
            "silhouette_score": self.silhouette_avg
        }

        self.save_clustering_results(clusters, cluster_labels, metrics, texts)

        self.visualize_clusters(embeddings, clusters, cluster_labels)
        
        return clusters, cluster_labels, metrics

def main():
    # Initialize clusterer
    clusterer = DocumentClusterer(num_clusters=8, batch_size=5)
    
    # Process dataset
    clusters, cluster_labels, metrics = clusterer.process_dataset(num_samples=80)
    
    # Print results
    print("\nClustering Results:")
    print(f"Runtime: {metrics['runtime_seconds']:.2f} seconds")
    print(f"Memory Usage: {metrics['memory_usage_mb']:.2f} MB")
    print("\nCluster Topics:")
    for cluster_id, terms in cluster_labels.items():
        print(f"Cluster {cluster_id}: {', '.join(terms)}")

if __name__ == "__main__":
    main()