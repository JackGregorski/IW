import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: for better separation
# from sklearn.manifold import TSNE
# import umap

# Load your encoding data (after parsing and validating it as you already do)
# Example: encoding_df is the DataFrame with shape (n_proteins, n_features)
#          cluster_labels is a list or array with cluster ID for each protein

def visualize_clusters(encoding_df, cluster_labels, method='pca', save_path=None):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'umap':
        import umap
        reducer = umap.UMAP(random_state=42)
    else:
        raise ValueError("Unsupported method. Choose 'pca', 'tsne', or 'umap'.")

    reduced = reducer.fit_transform(encoding_df)
    reduced_df = pd.DataFrame(reduced, columns=['x', 'y'])
    reduced_df['cluster'] = cluster_labels

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=reduced_df, x='x', y='y', hue='cluster', palette='tab10', s=50)
    plt.title(f"KMeans Clusters Visualized ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
