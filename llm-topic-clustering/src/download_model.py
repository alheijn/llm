from huggingface_hub import snapshot_download
from pathlib import Path

# Downloading the Mistral 7B model to models/mistral-7b folder
mistral_models_path = Path("models/mistral-7b")
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="mistralai/Mistral-7B-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)
