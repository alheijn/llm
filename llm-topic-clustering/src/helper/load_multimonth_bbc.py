from tqdm import tqdm
from datasets import load_dataset

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