#!/bin/bash

#SBATCH --job-name=Merge_Protein_Clusters
#SBATCH --output=/scratch/gpfs/jg9705/IW_logs/merge_clusters_%j.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_logs/merge_clusters_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate sn-torch-env

echo "Starting merge for all thresholded datasets..."

LOOKUP_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/protein_cluster_lookup.tsv"
INPUT_DIR="/scratch/gpfs/jg9705/IW_code/Model_Resources"
OUTPUT_DIR="/scratch/gpfs/jg9705/IW_code/Model_Resources/with_clusters"

mkdir -p "$OUTPUT_DIR"

for threshold in 400 500 600 700 800 900; do
    INPUT_FILE="${INPUT_DIR}/pairs_threshold${threshold}.tsv"
    OUTPUT_FILE="${OUTPUT_DIR}/pairs_threshold${threshold}_clustered.tsv"

    echo "→ Processing $INPUT_FILE"
    python add_clusters.py "$INPUT_FILE" "$LOOKUP_FILE" "$OUTPUT_FILE"
done

echo "✅ All thresholded datasets processed and saved to $OUTPUT_DIR"
