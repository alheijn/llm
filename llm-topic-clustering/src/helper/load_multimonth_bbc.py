from tqdm import tqdm
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
from helper.save_results import save_texts

def load_bbc_news_multimonth(months=['2024-10', '2024-11']):
    """Load BBC News data from multiple months"""
    
    all_texts = []
    
    for month in tqdm(months, desc="Loading BBC News datasets"):
        try:
            dataset = load_dataset('RealTimeData/bbc_news_alltime', month)
            data = dataset['train']
            
            texts = [item['content'] for item in data]
            
            all_texts.extend(texts)
            
        except Exception as e:
            print(f"Error loading dataset for {month}: {e}")
            continue
        
    print(f"number of bbc texts: {len(all_texts)}")
    
    return all_texts

def load_preprocessed_data(dataset_name, num_samples, output_dir):
    """Load and preprocess dataset"""
    import random
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
            save_texts(texts, output_dir, categories=categories if dataset_name == "20newsgroups" else None)            
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
                save_texts(texts, output_dir)
                
        except Exception as e:
            print(f"Error loading BBC News dataset: {e}")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return texts