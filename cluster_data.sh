#!/bin/bash

#SBATCH --job-name=Protein_Clustering
#SBATCH --output=/scratch/gpfs/jg9705/IW_data/logs/protein_clustering.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_data/logs/protein_clustering.err
#SBATCH --time=00:02:00      # Increased time limit
#SBATCH --mem=1G            # More memory for GPU processing
#SBATCH --cpus-per-task=4    # Keep 4 CPUs for preprocessing
#SBATCH --gres=gpu:1         # Request 1 GPU
#SBATCH --partition=GPU      # Use GPU partition
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
module load cuda/11.8   # Load CUDA

conda activate sn-torch-env

echo "Starting GPU-accelerated protein sequence clustering..."

# Run the modified GPU-enabled clustering script
python cluster_data.py --input /scratch/gpfs/jg9705/IW_data/gen_protein_encodings/encodings/encoding.tsv --output /scratch/gpfs/jg9705/IW_data/clustered_data/clustered_proteins.tsv --use_gpu

echo "Protein clustering completed successfully!"
