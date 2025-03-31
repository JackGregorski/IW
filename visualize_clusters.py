import pandas as pd
import argparse
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_clusters(encoding_file, cluster_file, method='pca', save_path=None):
    # Load protein encodings (no headers, 3 columns)
    encodings_df = pd.read_csv(encoding_file, sep='\t', header=None, dtype={2: str})
    encodings_df.columns = ['protein', 'sequence', 'encoding']
    encodings_df = encodings_df.dropna(subset=['encoding'])

    # Parse encoding into list of floats
    encoding_vectors = encodings_df['encoding'].apply(
        lambda x: list(map(float, x.strip().split())) if isinstance(x, str) else []
    )
    encoding_df = pd.DataFrame(encoding_vectors.tolist())
    encoding_df['protein'] = encodings_df['protein'].values

    # Load cluster labels (with headers: protein, protein_cluster)
    clusters_df = pd.read_csv(cluster_file, sep='\t')

    # Merge on protein
    merged_df = encoding_df.merge(clusters_df, on='protein', how='inner')

    # Separate data and labels
    X = merged_df.drop(columns=['protein', 'protein_cluster'])
    y = merged_df['protein_cluster']

    # Choose dimensionality reduction method
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

    reduced = reducer.fit_transform(X)
    reduced_df = pd.DataFrame(reduced, columns=['x', 'y'])
    reduced_df['cluster'] = y.values

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize protein clusters with PCA/TSNE/UMAP.")
    parser.add_argument("encoding_file", help="Path to encoding .tsv (no headers: protein, sequence, encoding)")
    parser.add_argument("cluster_file", help="Path to cluster assignment .tsv (with protein and protein_cluster columns)")
    parser.add_argument("save_path", help="Path to save output plot (e.g. output.png)")
    parser.add_argument("--method", choices=["pca", "tsne", "umap"], default="pca", help="Dimensionality reduction method")

    args = parser.parse_args()
    visualize_clusters(args.encoding_file, args.cluster_file, args.method, args.save_path)
