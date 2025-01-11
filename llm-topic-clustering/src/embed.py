from transformers import AutoTokenizer, AutoModel
import torch

class Embedder:
    def __init__(self, model_name='mistral-7b'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def generate_embeddings(self, documents, batch_size=32):
        embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                embeddings.append(batch_embeddings)
        return torch.cat(embeddings, dim=0)  # Concatenate all embeddings

def main():
    # Example usage
    embedder = Embedder()
    sample_documents = ["This is a sample document.", "Another document for testing."]
    embeddings = embedder.generate_embeddings(sample_documents)
    print(embeddings)

if __name__ == "__main__":
    main()