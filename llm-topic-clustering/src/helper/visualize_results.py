from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from collections import Counter
from sklearn.decomposition import PCA

def visualize_clusters(embeddings, clusters, cluster_labels, output_dir, num_clusters):
    """
    Create various visualizations for the clustering results.

    Args:
        embeddings (array): Document embeddings
        clusters (array): Cluster assignments
        cluster_labels (dict): Dictionary of cluster labels
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 10))

    # 1. PCA visualization of clusters
    ax1 = fig.add_subplot(121)
    _plot_cluster_pca(embeddings, clusters, cluster_labels, ax1, num_clusters)

    # 2. Cluster size distribution
    ax2 = fig.add_subplot(122)
    _plot_cluster_sizes(clusters, cluster_labels, ax2)

    plt.tight_layout()
    # save combined plot
    plt.savefig(
        os.path.join(output_dir, "plots", f"clusters_overview_{timestamp}.png")
    )
    plt.show()

    # 3. Term importance heatmap (separate figure)
    _plot_term_heatmap(clusters, cluster_labels, timestamp, num_clusters, output_dir)


def _plot_cluster_pca(embeddings, clusters, cluster_labels, ax, num_clusters):
    """Plot PCA projection of document embeddings"""
    # Reduce dimensionality to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Create scatter plot
    scatter = ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap="viridis", alpha=0.6
    )

    # Add cluster centers
    for i in range(num_clusters):
        mask = clusters == i
        if np.any(mask):
            center = embeddings_2d[mask].mean(axis=0)
            if i in cluster_labels:
                ax.annotate(
                    f"Cluster {i}\n({', '.join(cluster_labels[i][:2])})",
                    xy=center,
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                    arrowprops=dict(arrowstyle="->"),
                )

    ax.set_title("Document Clusters (PCA Projection)")
    ax.set_xlabel(f"PC1 (Variance: {pca.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 (Variance: {pca.explained_variance_ratio_[1]:.2%})")


def _plot_cluster_sizes(clusters, cluster_labels, ax):
    """Plot distribution of cluster sizes"""
    # Count documents per cluster
    cluster_sizes = Counter(clusters)

    # Create bars
    clusters_idx = list(cluster_sizes.keys())
    sizes = list(cluster_sizes.values())

    # Plot horizontal bars
    y_pos = np.arange(len(clusters_idx))
    bars = ax.barh(y_pos, sizes)

    # Customize plot
    ax.set_yticks(y_pos)
    labels = [
        f"Cluster {i}\n({', '.join(cluster_labels[i][:2])})" for i in clusters_idx
    ]
    ax.set_yticklabels(labels)


    # Add value labels on bars
    for i, v in enumerate(sizes):
        ax.text(v + 1, i, str(v), va="center")

    ax.set_title("Cluster Size Distribution")
    ax.set_xlabel("Number of Documents")


def _plot_term_heatmap(clusters, cluster_labels, timestamp, num_clusters, output_dir):
    """Create heatmap of term importance across clusters"""
    # Create term importance matrix
    terms = set()
    for terms_list in cluster_labels.values():
        terms.update(terms_list)
    terms = sorted(terms)

    # Create matrix of term presence (1) or absence (0)
    matrix = np.zeros((len(cluster_labels), len(terms)))
    for i, cluster_terms in cluster_labels.items():
        for term in cluster_terms:
            matrix[i, terms.index(term)] = 1

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        matrix,
        xticklabels=terms,
        yticklabels=[f"Cluster {i}" for i in range(num_clusters)],
        cmap="YlOrRd",
    )
    plt.title("Term Importance Across Clusters")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", f"term_heatmap_{timestamp}.png"))
    plt.show()
