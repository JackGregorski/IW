import pandas as pd
import cupy as cp  # GPU-accelerated NumPy alternative
from cuml.cluster import DBSCAN  # Faster clustering
from Bio import pairwise2
import torch
from joblib import Parallel, delayed

# Define output path
output_dir = "/scratch/gpfs/jg9705/IW_data/clustered_proteins"
output_file = f"{output_dir}/clustered_proteins.tsv"

# Load the .tsv file
df = pd.read_csv("/scratch/gpfs/jg9705/IW_data/gen_protein_encodings/encodings/encodings.tsv", sep="\t")

# Move data to GPU
df_values = df.iloc[:, 1].values  # Extract sequence column

# Compute similarity using CUDA-based alignment (Parallelized)
def compute_similarity(i, j):
    """Calculate sequence similarity using global alignment."""
    alignment = pairwise2.align.globalxx(df_values[i], df_values[j], score_only=True)
    return i, j, alignment

num_proteins = len(df)
similarity_matrix = cp.zeros((num_proteins, num_proteins), dtype=cp.float32)

# Parallelize similarity computation using Joblib
results = Parallel(n_jobs=4)(
    delayed(compute_similarity)(i, j)
    for i in range(num_proteins) for j in range(i+1, num_proteins)
)

# Store results in the similarity matrix
for i, j, sim_score in results:
    similarity_matrix[i, j] = similarity_matrix[j, i] = sim_score

# Convert similarity to distance (1 - normalized similarity)
max_sim = cp.max(similarity_matrix)
distance_matrix = 1 - (similarity_matrix / max_sim)

# Convert to float32 for cuML
distance_matrix = distance_matrix.astype(cp.float32)

# Perform clustering using GPU-accelerated DBSCAN
cluster_model = DBSCAN(eps=0.2, min_samples=2, metric="precomputed")
clusters = cluster_model.fit_predict(distance_matrix)

# Move results back to CPU
df["Cluster_ID"] = cp.asnumpy(clusters)

# Save results
df.to_csv(output_file, sep="\t", index=False)

print(f"Clustering complete! Results saved to {output_file}")
