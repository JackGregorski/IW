import pandas as pd
import argparse
from sklearn.cluster import KMeans

def cluster_proteins(encoding_file, pair_file, output_file, num_clusters=10):
    # Load protein encodings without headers and assign custom column names
    encodings_df = pd.read_csv(encoding_file, sep='\t', header=None, dtype={2: str})
    encodings_df.columns = ['protein', 'sequence', 'encoding']

    # Extract protein IDs and parse encoding column (assumes it's stored as space/comma-separated string)
    protein_ids = encodings_df['protein']

    # Split the encoding string into separate dimensions
    encoding_vectors = encodings_df['encoding'].apply(lambda x: list(map(float, x.strip().split())))
    encoding_df = pd.DataFrame(encoding_vectors.tolist())

    # Cluster using KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(encoding_df)

    # Create DataFrame with cluster labels
    protein_to_cluster = pd.DataFrame({
        'protein': protein_ids,
        'protein_cluster': cluster_labels
    })

    # Load positive/negative interaction pairs
    pairs_df = pd.read_csv(pair_file, sep='\t')

    # Merge clusters into the interaction data
    merged_df = pairs_df.merge(protein_to_cluster, on='protein', how='left')

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
