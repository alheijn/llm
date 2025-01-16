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
import string
import nltk
from helper.save_results import save_texts, save_clustering_results
from helper.visualize_results import visualize_clusters
import cluster_summarizer
from datasets import load_dataset
from helper.load_multimonth_bbc import load_bbc_news_multimonth
import random
import spacy
from collections import defaultdict


class DocumentClusterer:
    # def __init__(self, model_id="mistralai/Mistral-7B-v0.3", num_clusters=5, batch_size=5):
    def __init__(self, model_id="sshleifer/distilbart-cnn-12-6", num_clusters=5, batch_size=5):

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
            
        # load spaCy model for Named Entity Recognition
        self.nlp = spacy.load("en_core_web_sm")

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
                
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})        
            
            # Load model with float16 on MPS for better memory efficiency
            # self.model = AutoModelForCausalLM.from_pretrained(
            self.model = AutoModelForSeq2SeqLM.from_pretrained(                
                self.model_id,
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                low_cpu_mem_usage=True
            )#.resize_token_embeddings(len(self.tokenizer))
            
            # Move model to appropriate device
            self.model.to(self.device)
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"\nError loading model: {str(e)}")
            raise
        
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # convert to lowercase
        text = text.lower()
        
        phrases_to_remove = ['bbc', 'said', 'says', 'told', 'according to', 'reported', 'mr',
                       'This video can not be played To play this video you need to enable JavaScript in your browser.',
                       '\n', '\"', '\u201c', '\u201d', '\u2019s', r'\u00a', r'\u00d3' ]
        for phrase in phrases_to_remove:
            text = text.replace(phrase, '')
        
            
        return text
        
    def extract_named_entities(self, texts, important_types={'PERSON', 'ORG', 'GPE', 'EVENT', 'FAC', 'PRODUCT'}):
        '''Extract named entities from a list of texts'''
        entities_by_text = []
        
        for text in texts:
            doc = self.nlp(text[:100000])   # limit text length to avoid memory issues
            
            # extract entities of important types
            text_entities = defaultdict(list)
            for ent in doc.ents:
                if ent.label_ in important_types:
                    text_entities[ent.label_].append(ent.text)
            
            entities_by_text.append(dict(text_entities))
        
        #print(f"DEBUG: Entities for first text: {entities_by_text[0]}")
        return entities_by_text
        
        
    def get_embeddings(self, texts, show_progress=True):
        """Generate embeddings for texts in batches"""
        
        # extract named entities for all texts
        entities_by_text = self.extract_named_entities(texts)
        
        # combine original text with entities
        combined_texts = []
        for i, entities in enumerate(entities_by_text):
            text = texts[i]
            entity_text = ' '.join(f"{k}: {', '.join(v)}" for k, v in entities.items())
            combined_texts.append(f"{text}\n{entity_text}")
            
        embeddings = []
        iterator = tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings: ") if show_progress else range(0, len(texts), self.batch_size)
        for i in iterator:
            batch_texts = combined_texts[i:i + self.batch_size]
            # Tokenize with padding
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use the mean of the last hidden state as the embedding
                batch_embeddings = outputs.encoder_last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())        

            # Clear GPU/MPS memory after each batch
            if self.device == "mps":
                torch.mps.empty_cache()
        
        return np.vstack(embeddings)
        # return embeddings
        
    def cluster_documents(self, embeddings):
        """Perform K-means clustering with the best number of clusters based on silhouette score"""
        
        best_num_clusters = self.num_clusters
        best_silhouette_score = -1
        best_clusters = None
        best_kmeans = None
        
        for n_clusters in range(self.num_clusters - int(2*self.num_clusters/3), self.num_clusters + int(2*self.num_clusters/3)):
            if n_clusters < 2:
                continue
            
            print(f"Clustering with {n_clusters} clusters...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            silhouette_avg = silhouette_score(embeddings, clusters)
            print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.3f}")
            
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_num_clusters = n_clusters
                best_clusters = clusters
                best_kmeans = kmeans
        
        self.num_clusters = best_num_clusters
        self.silhouette_avg = best_silhouette_score
        print(f"Best number of clusters: {best_num_clusters} with Silhouette Score: {best_silhouette_score:.3f}")
        
        return best_clusters, best_kmeans
    
        # """Perform K-means clustering"""
                
        # print(f"Clustering {len(embeddings)} documents into {self.num_clusters} clusters...")
        # kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        # clusters = kmeans.fit_predict(embeddings)
        
        # # Calculate silhouette score
        # silhouette_avg = silhouette_score(embeddings, clusters)
        # self.silhouette_avg = silhouette_avg
        # print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        
        # self.calinski_harabasz = calinski_harabasz_score(embeddings, clusters)
        # print(f"CALINSKI HARABASZ Score: {silhouette_avg:.3f}")

        # return clusters, kmeans
        
    def get_cluster_labels(self, texts, clusters, kmeans):
        """Extract representative terms for each cluster using TF-IDF"""
        tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words=list(ENGLISH_STOP_WORDS),
            ngram_range=(1, 3),     # Include both single words and bigrams
            min_df=1,       # Term must appear in at least min_df documents
            max_df=0.90     # Ignore terms that appear in more than max_df% of documents
        )

        tfidf_matrix = tfidf.fit_transform(texts)
        feature_names = tfidf.get_feature_names_out()
        cluster_labels = {}
        
        # extract named entities for all texts
        all_entities = self.extract_named_entities(texts)
    
        for i in range(self.num_clusters):
            # Get texts in this cluster
            cluster_docs = tfidf_matrix[clusters == i]
            if cluster_docs.shape[0] == 0:
                cluster_labels[i] = ["Empty cluster"]
                continue
            
            # collect entities for this cluster
            cluster_entities = defaultdict(int)
            for idx in np.where(clusters == i)[0]:
                doc_entities = all_entities[idx]
                for entity_type, entity_list in doc_entities.items():
                    for entity in entity_list:
                        cluster_entities[entity] += 1
                        
            #print(f"DEBUG: Cluster {i} entities: {cluster_entities}")
            

            # Calculate mean TF-IDF scores for the cluster
            avg_tfidf = cluster_docs.mean(axis=0).A1
            
            # Get candidate terms
            top_indices = avg_tfidf.argsort()[-20:][::-1]  # Get more terms initially
            candidates = [(feature_names[idx], avg_tfidf[idx]) for idx in top_indices]  # store term and its tf-idf score
            top_terms = []
            
            # add most frequent entities to top terms first
            sorted_entities = sorted(
                cluster_entities.items(), key=lambda x: x[1], reverse=True
            )
            for entity, count in sorted_entities[:3]:
                if count > 1:
                    top_terms.append(entity)

            # Filter and group terms
            unigrams = []
            phrases = []

            for term, score in candidates:
                if term not in top_terms:
                    # Separate unigrams and phrases
                    if ' ' in term and score > 0.05:  # Higher threshold for phrases
                        phrases.append(term)
                    elif score > 0.025:  # Lower threshold for single words
                        unigrams.append(term)
            
                # Stop if we have enough terms
                if len(phrases) >= 2 and len(unigrams) >= 3:
                    break
        
            # Combine phrases and unigrams for final labels
            top_terms.extend(phrases[:2])
            top_terms.extend(unigrams[:3])
            
            # Ensure we have at least some terms
            if not top_terms and candidates:
                top_terms = [term for term, _ in candidates[:5]]
            print(f"DEBUG: Cluster {i}: {', '.join(top_terms)}")
            
            cluster_labels[i] = top_terms                   
            
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
                save_texts(texts, self.output_dir, categories=categories if dataset_name == "20newsgroups" else None)            
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
                    save_texts(texts, self.output_dir)
                
            except Exception as e:
                print(f"Error loading BBC News dataset: {e}")
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        
        # Track runtime and memory
        start_time = time.time()

        # Preprocess all texts
        print("\nDEBUG: start preprocessing")
        texts = [self.preprocess_text(text) for text in texts]

        # Generate embeddings
        print("\nDEBUG: start generating embeddings")
        embeddings = self.get_embeddings(texts)
        
        # Perform clustering
        print("\nDEBUG: start clustering")        
        clusters, kmeans = self.cluster_documents(embeddings)
        
        # Get cluster labels using TF-IDF
        cluster_labels = self.get_cluster_labels(texts, clusters, kmeans)
        print("TF-IDF Cluster Topics:")
        for cluster_id, terms in cluster_labels.items():
            print(f"Cluster {cluster_id}: {', '.join(terms)}")
            
        # generate and save summaries using efficient approach with TF-IDF labels
        print("\nGenerating cluster summaries...")
        cluster_summaries = self.generate_summaries(texts, clusters, cluster_labels)
        self.save_efficient_summaries(cluster_summaries)
        
        # Print summaries
        print("\nCluster Summaries:")
        for cluster_id, summary in cluster_summaries.items():
            print(f"\nCluster {cluster_id}:")
            print(f"Topic: {summary['topic']}")
            print(f"TF-IDF terms: {', '.join(summary['tfidf_terms'])}")
            #print(f"Key phrases: {', '.join(summary['key_phrases'])}")
            #print(f"Representative content: {summary['representative_sentence'][:200]}...")        
        
        # Calculate metrics
        end_time = time.time()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        metrics = {
            "runtime_seconds": end_time - start_time,
            "memory_usage_mb": final_memory - self.initial_memory,
            "silhouette_score": self.silhouette_avg
        }

        save_clustering_results(self.num_clusters, clusters, cluster_labels, metrics, texts, self.output_dir)

        visualize_clusters(embeddings, clusters, cluster_labels, self.output_dir, self.num_clusters)
        
        return clusters, cluster_labels, metrics, self.silhouette_avg

    # def generate_summaries(self, texts, clusters, cluster_labels):
    def generate_summaries(self, texts, clusters, cluster_labels):
        '''Generate summaries for each cluster'''
        cluster_summaries = {}
        
        for cluster_id in tqdm(range(self.num_clusters), desc="Analyzing clusters"):
            # get texts for this cluster
            cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster_id]
            
            if not cluster_texts:
                cluster_summaries[cluster_id] = {
                    'topic': "Empty cluster",
                    'tfidf_terms': [],
                    'example_texts': []
                }
                continue
            
            selected_texts = self.summarizer.get_representative_texts(
                cluster_texts=cluster_texts, cluster_labels=cluster_labels, cluster_id=cluster_id)
                
            combined_text = "\n---\n".join(selected_texts)
            
            # Create prompt
            prompt = (
                "Give a list of maximum 10 words that represent the topics of the cluster.:\n\n"
                #f"TF-IDF terms: {', '.join(cluster_labels[cluster_id])}.\n"
                f"Relevant cluster content: {combined_text}\n"
            )
            
            # Prepare input for model (DistilBART)
            inputs = self.tokenizer(
                prompt,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            ).to(self.device)
            
            print("DEBUG: Starting summary generation")
            
            # generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100, # control output length
                    num_return_sequences=1,
                    early_stopping=True,
                    num_beams=4,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
            topic_label = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # combine results
            summary = {
                'topic': topic_label,
                'tfidf_terms': cluster_labels[cluster_id],
                'example_texts': combined_text
                #'key_phrases': key_phrases[:5],
                #'representative_sentence': rep_sentences[0] if rep_sentences else ''
            }
            
            # print(f"DEBUG: {summary}")
            
            cluster_summaries[cluster_id] = summary
            
            # clear gpu/mps memory after each cluster
            if self.device.type == "mps":
                torch.mps.empty_cache()
                
        return cluster_summaries
    
    def save_efficient_summaries(self, cluster_summaries):
        """Save the efficient cluster summaries to a results file"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.output_dir, 'results', f'cluster_summaries_{timestamp}.txt')
        with open(summary_file, 'w') as f:
            for cluster_id, summary in cluster_summaries.items():
                f.write(f"\nCluster {cluster_id}:\n")
                f.write(f"Topic: {summary['topic']}\n")
                f.write(f"Example texts: {summary['example_texts']}\n")
                #f.write(f"Key phrases: {', '.join(summary['key_phrases'])}\n")
                #f.write(f"Representative content: {summary['representative_sentence']}\n")
                f.write("-" * 80 + "\n")

def main():
    # Initialize clusterer
    clusterer = DocumentClusterer(num_clusters=10, batch_size=5)
    
    # Process dataset
    clusters, cluster_labels, metrics, silhouette_avg = clusterer.process_dataset(num_samples=200)
    
    # Print results
    print("\nClustering Results:")
    print(f"Runtime: {metrics['runtime_seconds']:.2f} seconds")
    print(f"Memory Usage: {metrics['memory_usage_mb']:.2f} MB")
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print("\nCluster Topics:")
    for cluster_id, terms in cluster_labels.items():
        print(f"Cluster {cluster_id}: {', '.join(terms)}")

if __name__ == "__main__":
    main()