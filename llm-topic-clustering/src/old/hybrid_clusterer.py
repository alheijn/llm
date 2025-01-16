import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import numpy as np
import time
import psutil
from tqdm import tqdm
import platform
from sklearn.datasets import fetch_20newsgroups
import os
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from helper.save_results import save_texts, save_clustering_results
from helper.visualize_results import visualize_clusters
import cluster_summarizer
from datasets import load_dataset
from helper.load_multimonth_bbc import load_bbc_news_multimonth
import random
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import hdbscan
import nltk
from nltk.corpus import stopwords

class HybridClusterer:
    # def __init__(self, model_id="mistralai/Mistral-7B-v0.3", num_clusters=5, batch_size=5):
    # def __init__(self, model_id="sshleifer/distilbart-cnn-12-6", num_clusters=5, batch_size=5):
    def __init__(self, model_id='all-MiniLM-L6-v2', num_clusters=5, batch_size=5):

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
        
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        

        # create output directories to store the results
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        for subdir in ['texts', 'plots', 'results']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
            
        # initialize EfficientClusterSummarizer
        self.summarizer = cluster_summarizer.ClusterSummarizer()

    def setup_model(self):
        """Initialize the model optimized for Apple Silicon"""
        print(f"Loading model and tokenizer (using {self.device} device)...")
        
        try:
            # Set memory management for MPS device
            if self.device.type == "mps":
                # Set memory fraction to 0.0 to enable dynamic memory allocation
                torch.mps.set_per_process_memory_fraction(0.0)
                print("Configured MPS memory management")
                
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)            
            # # Add padding token if it doesn't exist
            # if self.tokenizer.pad_token is None:
            #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})        
            
            self.model = SentenceTransformer(self.model_id)
            
            # Move model to appropriate device
            self.model.to(self.device)
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"\nError loading model: {str(e)}")
            raise
        
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # convert to lowercase
        processed = text.lower()
        
        # remove stop words
        for word in ENGLISH_STOP_WORDS:
            processed = processed.replace(word, '')
            
        reporter_phrases = ['said', 'says', 'told', 'according to', 'reported']
        for phrase in reporter_phrases:
            processed = processed.replace(phrase, '')
            
        # Remove short words (length < 3)
        words = text.split()
        words = [w for w in words if len(w) >= 3 and w not in self.stop_words]      
          
        return processed
        

    def get_hybrid_embeddings(self, texts):
        """Combine TF-IDF and semantic embeddings for better clustering"""
        # First pass: TF-IDF on full corpus (needed for vocabulary)
        print("Generating TF-IDF features...")
        tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        tfidf_matrix = tfidf.fit_transform(texts)
        tfidf_dense = normalize(tfidf_matrix.toarray())
        
        # Second pass: Generate semantic embeddings in batches
        print("Generating semantic embeddings in batches...")
        num_texts = len(texts)
        semantic_embeddings = []
        
        for i in tqdm(range(0, num_texts, self.batch_size)):
            batch_texts = texts[i:min(i + self.batch_size, num_texts)]
            
            # we only do inference (forward pass) - no gradients for backpropagation needed
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    batch_size=self.batch_size
                )
                semantic_embeddings.append(batch_embeddings)
            
            # Clear GPU/MPS memory after each batch
            if self.device == "mps":
                torch.mps.empty_cache()
            
        # Combine all batches
        semantic_embeddings = np.vstack(semantic_embeddings)
        semantic_embeddings = normalize(semantic_embeddings)
        
        # Combine TF-IDF and semantic embeddings
        print("Combining embeddings...")
        combined = np.hstack([
            0.3 * tfidf_dense,
            0.7 * semantic_embeddings
        ])
        
        return combined


    def extract_cluster_topics(self, texts, cluster_labels):
        """Extract more meaningful topic labels for each cluster"""
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[idx])
            
        topics = {}
        for label, cluster_texts in clusters.items():
            # Get named entities and important phrases
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=1000,
                stop_words='english',
                #token_pattern=r'(?u)\b[A-Za-z][A-Za-z-]+[A-Za-z]\b'
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                feature_names = vectorizer.get_feature_names_out()
                mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
                
                # Get top phrases
                significant_terms = []
                for idx in mean_tfidf.argsort()[::-1]:
                    term = feature_names[idx]
                    significant_terms.append(term)
                    if len(significant_terms) == 10:  # Get top 5 significant terms
                        break
                        
                topics[label] = significant_terms
                
            except:
                topics[label] = ['unknown']
                
        return topics
    
    
    def evaluate_clusters(self, embeddings, labels):
        """Evaluate cluster quality with multiple metrics"""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        metrics = {
            'silhouette_score': silhouette_score(embeddings, labels),
            'calinski_harabasz': calinski_harabasz_score(embeddings, labels)
        }
        
        # Calculate cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = {f'cluster_{l}': c for l, c in zip(unique_labels, counts)}
        
        return metrics
        
        
    def cluster_documents(self, embeddings):
        """Perform HDBSCAN/K-means clustering"""
        print(f"Clustering {len(embeddings)} documents into {self.num_clusters} clusters...")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.num_clusters, 
            min_samples=None,
            metric='euclidean',
            cluster_selection_method='eom'  # "excess of mass"
        )
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Check if we got valid clustering
        unique_clusters = len(np.unique(cluster_labels))
        
        # If clustering failed, try KMeans as fallback
        if unique_clusters < 2:
            print("HDBSCAN produced single cluster, falling back to KMeans...")
            kmeans = KMeans(
                n_clusters=self.num_clusters,
                random_state=42
            )
            cluster_labels = kmeans.fit_predict(embeddings)
        
        ## OLD
        # kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        # clusters = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        self.silhouette_avg = silhouette_avg
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
        print(f"calinski harabasz score: {calinski_harabasz}")
        
        return cluster_labels

    def process_dataset(self, dataset_name="bbc_news_alltime", num_samples=2000):
        """Main processing pipeline"""
        # Load and preprocess dataset
        if dataset_name == "20newsgroups":
            dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
            # Sample if needed
            if num_samples and num_samples < len(dataset.data):
                random.seed(42)
                indices = random.sample(range(len(dataset.data)), num_samples)
                texts = [dataset.data[i] for i in indices]
                categories = [dataset.target_names[dataset.target[i]] for i in indices]
                print(f"\nSampled categories: {set(categories)}")    

                # Save input texts
                save_texts(texts, categories=categories if dataset_name == "20newsgroups" else None)            
            else:
                texts = dataset.data
        elif dataset_name == "bbc_news_alltime":
            # https://huggingface.co/datasets/RealTimeData/bbc_news_alltime
            try:
                texts = load_bbc_news_multimonth()
                
                # Sample if needed
                if num_samples and num_samples < len(texts):
                    random.seed(42)
                    indices = random.sample(range(len(texts)), num_samples)
                    texts = [texts[i] for i in indices]                    
                    # Save input texts
                    save_texts(texts)
                
            except Exception as e:
                print(f"Error loading BBC News dataset: {e}")
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        
        # Track runtime and memory
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Preprocess all texts
        texts = [self.preprocess_text(text) for text in texts]

        # Generate embeddings
        embeddings = self.get_hybrid_embeddings(texts)
        
        # Perform clustering
        cluster_labels = self.cluster_documents(embeddings)
        
        # Get cluster labels using TF-IDF
        topics = self.extract_cluster_topics(texts, cluster_labels)
        print("TF-IDF Cluster Topics:")
        for cluster_id, terms in topics.items():
            print(f"Cluster {cluster_id}: {', '.join(terms)}")
            
        # # generate and save summaries using efficient approach with TF-IDF labels
        # print("\nGenerating cluster summaries...")
        # cluster_summaries = self.generate_summaries(texts, cluster_labels, cluster_labels)
        # self.save_efficient_summaries(cluster_summaries)
        # 
        # # Print summaries
        # print("\nCluster Summaries:")
        # for cluster_id, summary in cluster_summaries.items():
        #     print(f"\nCluster {cluster_id}:")
        #     print(f"Topic: {summary['topic']}")
        #     print(f"TF-IDF terms: {', '.join(summary['tfidf_terms'])}")
        #     #print(f"Key phrases: {', '.join(summary['key_phrases'])}")
        #     #print(f"Representative content: {summary['representative_sentence'][:200]}...")        
        
        # Calculate metrics
        end_time = time.time()
        runtime = time.time() - start_time        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        metrics = self.evaluate_clusters(embeddings, cluster_labels)
        
        # metrics = {
        #     "runtime_seconds": end_time - start_time,
        #     "memory_usage_mb": final_memory - self.initial_memory,
        #     "silhouette_score": self.silhouette_avg
        # }
        # Prepare results
        results = {
            'cluster_labels': cluster_labels,
            'topics': topics,
            'metrics': metrics,
            'performance': {
                'runtime_seconds': runtime,
                'memory_usage_mb': memory_used
            }
        }

        visualize_clusters(embeddings, cluster_labels, topics, self.output_dir, self.num_clusters)

        save_clustering_results(self.num_clusters, cluster_labels, topics, metrics, texts)

        
        self._print_clustering_summary(results)        
        
        # return cluster_labels, cluster_labels, metrics, self.silhouette_avg
        return results
    
    
    def _print_clustering_summary(self, results):
        """Print a summary of clustering results"""
        print("\nClustering Results:")
        print(f"Runtime: {results['performance']['runtime_seconds']:.2f} seconds")
        print(f"Memory Usage: {results['performance']['memory_usage_mb']:.2f} MB")
        print(f"Silhouette Score: {results['metrics']['silhouette']:.3f}")
        
        print("\nCluster Topics:")
        for cluster_id, terms in results['topics'].items():
            if cluster_id != -1:  # -1 is noise in HDBSCAN
                print(f"Cluster {cluster_id}: {', '.join(terms)}")
        
        if -1 in results['topics']:
            n_noise = len([l for l in results['cluster_labels'] if l == -1])
            print(f"\nNoise points (unclustered): {n_noise}")    

def main():
    # Initialize clusterer
    clusterer = HybridClusterer(num_clusters=5, batch_size=5)
    
    # Process dataset
    # clusters, cluster_labels, metrics, silhouette_avg = clusterer.process_dataset(num_samples=1000)
    clusterer.process_dataset(num_samples=200)
    

if __name__ == "__main__":
    main()