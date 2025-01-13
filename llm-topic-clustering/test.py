from huggingface_hub import list_datasets
datasets_list = list_datasets()
print(', '.join(dataset.id for dataset in datasets_list))