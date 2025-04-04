#!/bin/bash

#SBATCH --job-name=Cluster_Proteins
#SBATCH --output=/scratch/gpfs/jg9705/IW_data/logs/cluster_proteins.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_data/logs/cluster_proteins.err
#SBATCH --time=01:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate sn-torch-env

echo "Starting protein clustering..."

# Define input/output file paths
ENCODING_FILE="/scratch/gpfs/jg9705/IW_data/gen_protein_encodings/encodings/encodings.tsv"
PAIR_FILE="/scratch/gpfs/jg9705/IW_data/Model_Resources/positives.tsv"
OUTPUT_FILE="/scratch/gpfs/jg9705/IW_data/Model_Resources/pairs_with_clusters.tsv"
NUM_CLUSTERS=1000

# Run the clustering script
python cluster_kmeans.py "$ENCODING_FILE" "$PAIR_FILE" "$OUTPUT_FILE" --num_clusters "$NUM_CLUSTERS"

echo "Protein clustering completed!"
