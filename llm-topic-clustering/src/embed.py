import torch
from transformers import AutoModel, AutoTokenizer

# Load the model and tokenizer
model_name = "mistralai/Mistral-7B-v0.3"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts using a pre-trained model.

    Args:
        texts (list of str): A list of text strings to generate embeddings for.

    Returns:
        list of numpy.ndarray: A list of embeddings, where each embedding is a numpy array.
    """
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
    return embeddings