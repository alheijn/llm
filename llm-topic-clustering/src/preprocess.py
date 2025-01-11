import re
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_texts(texts):
    """
    Preprocess a list of texts by removing special characters, converting to lowercase,
    tokenizing, lemmatizing, and removing stopwords.

    Args:
        texts (list of str): A list of text strings to be preprocessed.

    Returns:
        list of str: A list of preprocessed text strings.
    """

    preprocessed_texts = []
    for text in texts:
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()

        # Tokenize and lemmatize the text, removing stopwords
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop]
        preprocessed_texts.append(' '.join(tokens))
    return preprocessed_texts