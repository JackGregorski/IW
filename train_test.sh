#!/bin/bash

#SBATCH --job-name=Split_All_Thresholds
#SBATCH --output=/scratch/gpfs/jg9705/IW_data/logs/split_by_cluster_%j.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_data/logs/split_by_cluster_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate jg-torch-env

echo "Starting train/test splits for all thresholds..."

INPUT_DIR="/scratch/gpfs/jg9705/IW_code/Model_Resources/with_clusters"
OUTPUT_BASE_DIR="/scratch/gpfs/jg9705/IW_code/Model_Resources/splits"

for threshold in 400 500 600 700 800 900; do
    INPUT_FILE="${INPUT_DIR}/pairs_threshold${threshold}_clustered.tsv"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/threshold${threshold}"

    echo "→ Processing $INPUT_FILE"
    python train_test_val_split.py "$INPUT_FILE" "$OUTPUT_DIR" --test_frac 0.2 --toy_frac 0.0 --seed 42
done

echo "✅ All datasets split and saved under $OUTPUT_BASE_DIR"
