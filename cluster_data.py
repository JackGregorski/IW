import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from Bio import pairwise2
import os

# Define output path
output_dir = "/scratch/gpfs/jg9705/IW_data/clustered_proteins"
output_file = os.path.join(output_dir, "clustered_proteins.tsv")

# Step 1: Load the .tsv file
df = pd.read_csv("/scratch/gpfs/jg9705/IW_data/gen_protein_encodings/encodings/encodings.tsv", sep="\t")

# Step 2: Compute similarity (pairwise alignment scores)
def compute_similarity(seq1, seq2):
    """Calculate sequence similarity using pairwise alignment."""
    alignment = pairwise2.align.globalxx(seq1, seq2, score_only=True)
    return alignment

# Create similarity matrix
num_proteins = len(df)
similarity_matrix = np.zeros((num_proteins, num_proteins))

for i in range(num_proteins):
    for j in range(i+1, num_proteins):
        sim_score = compute_similarity(df.iloc[i, 1], df.iloc[j, 1])  # Fixed `.iloc`
        similarity_matrix[i, j] = similarity_matrix[j, i] = sim_score

# Convert similarity to distance (1 - normalized similarity)
max_sim = np.max(similarity_matrix)
distance_matrix = 1 - (similarity_matrix / max_sim)

# Step 3: Perform clustering (Hierarchical Clustering)
cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.2, linkage="average", metric="precomputed")
clusters = cluster_model.fit_predict(distance_matrix)

# Step 4: Save results
df["Cluster_ID"] = clusters
df.to_csv(output_file, sep="\t", index=False)

print(f"Clustering complete! Results saved to {output_file}")
