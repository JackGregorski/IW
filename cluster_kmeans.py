import pandas as pd
import argparse
from sklearn.cluster import KMeans

def cluster_unique_proteins(encoding_file, protein_list_file, output_file, num_clusters=10):
    # Load all protein encodings
    encodings_df = pd.read_csv(encoding_file, sep='\t', header=None)
    encodings_df.columns = ['protein', 'encoding']
    encodings_df = encodings_df.dropna(subset=['encoding'])

    # Load unique protein IDs to cluster
    unique_prots_df = pd.read_csv(protein_list_file, sep='\t', header=None)
    unique_prots_df.columns = ['protein']
    unique_proteins = set(unique_prots_df['protein'])

    # Filter encodings to just the unique proteins
    filtered_df = encodings_df[encodings_df['protein'].isin(unique_proteins)].copy()

    # Convert encodings to numeric arrays
    encoding_vectors = filtered_df['encoding'].apply(lambda x: list(map(float, x.strip().split())))
    encoding_matrix = pd.DataFrame(encoding_vectors.tolist())
    valid_rows = encoding_matrix.dropna()

    filtered_proteins = filtered_df.iloc[valid_rows.index]['protein'].reset_index(drop=True)
    encoding_matrix = valid_rows.reset_index(drop=True)

    # Cluster encodings
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(encoding_matrix)

    # Save mapping
    cluster_map = pd.DataFrame({
        'protein': filtered_proteins,
        'protein_cluster': cluster_labels
    })
    cluster_map.to_csv(output_file, sep='\t', index=False)
    print(f"Saved protein-cluster lookup table to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster unique proteins and save mapping.")
    parser.add_argument("encoding_file", help="Protein encoding file (3-column TSV)")
    parser.add_argument("protein_list_file", help="List of unique proteins to cluster (1-column TSV)")
    parser.add_argument("output_file", help="Path to save protein-cluster lookup TSV")
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of clusters (default: 10)")

    args = parser.parse_args()
    cluster_unique_proteins(args.encoding_file, args.protein_list_file, args.output_file, args.num_clusters)
