import json
import os
from datetime import datetime
import numpy as np

def save_texts(self, texts, categories=None):
    """Save input texts and their categories"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(self.output_dir, 'texts', f'input_texts_{timestamp}.json')
        
    data = {
        'texts': texts,
        'categories': categories if categories else []
    }
        
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def save_clustering_results(self, clusters, cluster_labels, metrics, texts):
    """Save clustering results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(self.output_dir, 'results', f'clustering_results_{timestamp}.json')
        
    # Convert numpy types to native Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = {
        'num_clusters': self.num_clusters,
        'silhouette_score': float(metrics.get('silhouette_score', 0)),
        'runtime_seconds': float(metrics.get('runtime_seconds', 0)),
        'memory_usage_mb': float(metrics.get('memory_usage_mb', 0)),
        'cluster_labels': {str(k): v for k, v in cluster_labels.items()},
        'cluster_assignments': {
            str(i): {
                'cluster': int(clusters[i]),
                'text': texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i]
            }
            for i in range(len(texts))
        }
    }
        
    # Convert all values to ensure serialization
    results = {k: convert_to_serializable(v) for k, v in results.items()}
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)