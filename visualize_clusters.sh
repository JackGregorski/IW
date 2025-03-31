#!/bin/bash

#SBATCH --job-name=Visualize_Clusters
#SBATCH --output=/scratch/gpfs/jg9705/IW_data/logs/visualize_clusters.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_data/logs/visualize_clusters.err
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate sn-torch-env

echo "Starting cluster visualization..."

# Set variables
ENCODING_FILE="/scratch/gpfs/jg9705/IW_data/gen_protein_encodings/encodings/encodings.tsv"
CLUSTER_FILE="/scratch/gpfs/jg9705/IW_data/Model_Resources/pairs_with_clusters.tsv"
METHOD="pca"  # or tsne or umap
SAVE_PATH="/scratch/gpfs/jg9705/IW_data/kmeans_clusters_${METHOD}.png"

# Run the visualization script with proper argument order
python visualize_clusters.py "$ENCODING_FILE" "$CLUSTER_FILE" "$SAVE_PATH" --method "$METHOD"

echo "Cluster visualization completed and saved to $SAVE_PATH"
