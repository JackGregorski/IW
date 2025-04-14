#!/bin/bash

#SBATCH --job-name=Split_By_Cluster
#SBATCH --output=/scratch/gpfs/jg9705/IW_data/logs/split_by_cluster.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_data/logs/split_by_cluster.err
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate sn-torch-env

echo "Starting cluster-based dataset split..."

# Input TSV and output directory
INPUT_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/positives.tsv"
OUTPUT_DIR="/scratch/gpfs/jg9705/IW_code/Model_Resources/splits"

# Run the Python script
python train_test_val_split.py "$INPUT_FILE" "$OUTPUT_DIR" --test_frac 0.2 --toy_frac 0.1 --seed 42

echo "Dataset split complete!"
