#!/bin/bash

#SBATCH --job-name=Cluster_UniqueProteins
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
conda activate jg-torch-env

echo "Starting protein clustering..."

ENCODING_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/encodings.tsv"
UNIQUE_PROTEINS_FILE="/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/unique_proteins.tsv"
OUTPUT_LOOKUP="/scratch/gpfs/jg9705/IW_code/Model_Resources/protein_cluster_lookup.tsv"
NUM_CLUSTERS=1000

python cluster_kmeans.py "$ENCODING_FILE" "$UNIQUE_PROTEINS_FILE" "$OUTPUT_LOOKUP" --num_clusters "$NUM_CLUSTERS"

echo "Protein clustering completed!"
