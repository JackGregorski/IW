import pandas as pd
import argparse
from sklearn.cluster import KMeans

def cluster_proteins(encoding_file, pair_file, output_file, num_clusters=10):
    # Load protein encodings without headers and assign custom column names
    encodings_df = pd.read_csv(encoding_file, sep='\t', header=None, dtype={2: str})
    encodings_df.columns = ['protein', 'sequence', 'encoding']

    # Drop rows with missing encodings
    encodings_df = encodings_df.dropna(subset=['encoding'])

    # Safely parse the space-separated encoding string into floats
    encoding_vectors = encodings_df['encoding'].apply(
        lambda x: list(map(float, x.strip().split())) if isinstance(x, str) else []
    )
    encoding_df = pd.DataFrame(encoding_vectors.tolist())
    valid_rows = encoding_df.dropna()

    # Align protein IDs with valid encoding rows
    protein_ids = encodings_df.iloc[valid_rows.index]['protein'].reset_index(drop=True)
    encoding_df = valid_rows.reset_index(drop=True)

    # Cluster using KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(encoding_df)

    # Create DataFrame mapping proteins to their clusters
    protein_to_cluster = pd.DataFrame({
        'protein': protein_ids,
        'protein_cluster': cluster_labels
    })

    # Load the positive/negative interaction data
    pairs_df = pd.read_csv(pair_file, sep='\t')

    # Merge cluster label into the interaction data
    merged_df = pairs_df.merge(protein_to_cluster, on='protein', how='left')

    # Drop any rows that don't have a protein cluster (no encoding available)
    merged_df = merged_df.dropna(subset=['protein_cluster'])

    # Save final output
    merged_df.to_csv(output_file, sep='\t', index=False)
    print(f"Clustered and merged data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster proteins and label pair data with protein cluster.")
    parser.add_argument("encoding_file", help="Path to the file with protein encodings (no headers, 3 columns)")
    parser.add_argument("pair_file", help="Path to the file with positive and negative pairs")
    parser.add_argument("output_file", help="Path to save the pair data with cluster labels")
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of clusters for KMeans (default: 10)")

    args = parser.parse_args()
    cluster_proteins(args.encoding_file, args.pair_file, args.output_file, args.num_clusters)
