import json
import os
from datetime import datetime
import numpy as np

def save_texts(texts, output_dir, categories=None):
    """Save input texts and their categories"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    #output_texts_dir = os.path.join(output_dir, 'texts')
    #os.makedirs(output_texts_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'texts', f'input_texts_{timestamp}.json')
        
    data = {
        'texts': texts,
        'categories': categories if categories else []
    }
        
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def save_clustering_results(num_clusters, clusters, cluster_labels, metrics, texts, output_dir):
    """Save clustering results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    output_path = os.path.join(output_dir, 'results', f'clustering_results_{timestamp}.json')
        
    # Convert numpy types to native Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    cluster_assignments = {}
    for i in range(len(texts)):
        cluster_id = int(clusters[i])
        if cluster_id not in cluster_assignments:
            cluster_assignments[cluster_id] = []
        cluster_assignments[cluster_id].append(
            texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i]
        )

    results = {
        'num_clusters': num_clusters,
        'silhouette_score': float(metrics.get('silhouette_score', 0)),
        'runtime_seconds': float(metrics.get('runtime_seconds', 0)),
        'memory_usage_mb': float(metrics.get('memory_usage_mb', 0)),
        'cluster_labels': {str(k): v for k, v in cluster_labels.items()},
        'cluster_assignments': [
            {'cluster_id': cluster_id, 'cluster_texts': texts}
            for cluster_id, texts in cluster_assignments.items()
        ]
    }
        
    # Convert all values to ensure serialization
    results = {k: convert_to_serializable(v) for k, v in results.items()}
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)