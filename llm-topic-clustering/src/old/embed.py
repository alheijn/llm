import torch
from transformers import AutoModel, AutoTokenizer

# Load the model and tokenizer
model_name = "mistralai/Mistral-7B-v0.3"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Enable gradient checkpointing for memory optimization (at expense of computational time)
# model.gradient_checkpointing_enable() # did not lead to any performance improvement, in fact it made the embedding slightly slower

# Check if MPS (Metal Performance Shaders backend) is available and set the device accordingly
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Set the per-process memory fraction for MPS
if device.type == "mps":
    torch.mps.set_per_process_memory_fraction(0.0)

try:
    model.to(device)
except RuntimeError as e:
    if "out of memory" in str(e) or "Placeholder storage has not been allocated on MPS device" in str(e):
        print("MPS backend out of memory or not properly allocated, switching to CPU")
        device = torch.device("cpu")
        model.to(device)
    else:
        raise e

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
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
        #inputs.to("mps" if device.type == "mps" else device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
    return embeddings