import pandas as pd
import matplotlib.pyplot as plt

# Load MMseqs2 clustering results
df = pd.read_csv("/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/clusters.tsv", sep="\t", header=None, names=["Protein_ID", "Cluster_Representative"])

# Count cluster sizes
cluster_sizes = df["Cluster_Representative"].value_counts()

# 📌 1. Histogram of Cluster Sizes
plt.figure(figsize=(10, 5))
plt.hist(cluster_sizes, bins=50, log=True, edgecolor='black')
plt.xlabel("Cluster Size (Number of Proteins per Cluster)")
plt.ylabel("Frequency (Log Scale)")
plt.title("Distribution of Protein Cluster Sizes")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# 📌 2. Bar Chart of Top 10 Largest Clusters
top_clusters = cluster_sizes.head(10)

plt.figure(figsize=(10, 5))
top_clusters.plot(kind="bar", color="blue", edgecolor="black")
plt.xlabel("Cluster Representative")
plt.ylabel("Number of Proteins")
plt.title("Top 10 Largest Protein Clusters")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# 📌 3. Table of Largest Clusters
largest_clusters_df = top_clusters.reset_index()
largest_clusters_df.columns = ["Cluster Representative", "Size"]
print("Top 10 Largest Clusters:")
print(largest_clusters_df)
