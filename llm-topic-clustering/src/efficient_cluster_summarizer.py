from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EfficientClusterSummarizer:
    def __init__(self):
        # initialize a smaller, more efficient model for sentence embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_representative_sentences(self, texts, n_sentences=3):
        '''Extract most representative sentences using sentence embeddings'''
        # # split texts into sentences and get embeddings
        # sentences = [sent.strip() for text in texts for sent in text.split('.') if len(sent.strip()) > 20]
        # if not sentences:
        #     return []
            # Split texts into sentences
        sentences = []
        for text in texts:
            # Clean up text first
            cleaned_text = ' '.join(text.split())  # Remove extra whitespace
            # Split into sentences and filter
            for sent in cleaned_text.split('.'):
                sent = sent.strip()
                # Only keep sentences that:
                # - Are between 50 and 200 characters
                # - Contain at least 3 words
                # - Don't start with common prefixes that might indicate quotes or metadata
                if (20 <= len(sent) <= 200 and 
                    len(sent.split()) >= 3 and
                    not any(sent.lower().startswith(prefix) for prefix in ['>', 'wrote:', 're:', 'subject:', 'from:', 'to:'])):
                    sentences.append(sent)
        
        if not sentences:
            return []
        
        embeddings = self.sentence_model.encode(sentences, show_progress_bar=True)
        # get centroid
        centroid = np.mean(embeddings, axis=0)
        # calculate distances to centroid
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        
        # # get indices of closest sentences
        # closest_indices = np.argsort(distances)[:n_sentences]
        #
        # return [sentences[i] for i in closest_indices]
        
        # Get indices of closest sentences that are sufficiently different from each other
        selected_indices = []
        selected_sentences = []
    
        for idx in np.argsort(distances):
            current_sent = sentences[idx]
            # Check if this sentence is sufficiently different from already selected ones
            if not selected_sentences or all(
                self.sentence_similarity(current_sent, selected) < 0.8 
                for selected in selected_sentences
            ):
                selected_indices.append(idx)
                selected_sentences.append(current_sent)
                if len(selected_sentences) == n_sentences:
                    break
                
        return selected_sentences
    
    def sentence_similarity(self, sent1, sent2):
        """Calculate similarity between two sentences"""
        emb1 = self.sentence_model.encode([sent1], show_progress_bar=False)
        emb2 = self.sentence_model.encode([sent2], show_progress_bar=False)
        return np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
    
    def extract_key_phrases(self, texts):
        '''Extract key phrases using TF-IDF'''
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=20,
            stop_words='english',
            token_pattern=r'(?u)\b[A-Za-z][A-Za-z-]+[A-Za-z]\b'  # Requires terms to be at least 3 chars
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            features = vectorizer.get_feature_names_out()
            scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # # get top phrases
            # top_indices = np.argsort(scores)[-5:][::-1]
            # return [features[i] for i in top_indices]

            # Get top phrases with scores
            phrase_scores = [(features[i], scores[i]) for i in range(len(features))]
            # Filter out low-scoring phrases and sort by score
            significant_phrases = [
                phrase for phrase, score in phrase_scores 
                if score > 0.1 and len(phrase) > 3  # Only keep meaningful phrases
            ]
            return significant_phrases[:5]  # Return top 5 significant phrases
        except:
            return []
        