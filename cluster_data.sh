#!/bin/bash

#SBATCH --job-name=Protein_Clustering
#SBATCH --output=/scratch/gpfs/jg9705/IW_data/logs/protein_clustering.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_data/logs/protein_clustering.err
#SBATCH --time=08:00:00      # Adjust based on dataset size
#SBATCH --mem=8G             # Increase memory for large datasets
#SBATCH --cpus-per-task=4    # Use multiple CPUs for efficiency
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate sn-torch-env

echo "Starting protein sequence clustering..."

# Run the clustering script
python cluster_data.py --input /scratch/gpfs/jg9705/IW_data/gen_protein_encodings/encodings/encoding.tsv --output /scratch/gpfs/jg9705/IW_data/clustered_data/clustered_proteins.tsv

echo "Protein clustering completed successfully!"
