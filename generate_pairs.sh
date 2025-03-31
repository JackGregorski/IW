#!/bin/bash

#SBATCH --job-name=Filter_PosNeg
#SBATCH --output=/scratch/gpfs/jg9705/IW_data/logs/filter_pos_neg.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_data/logs/filter_pos_neg.err
#SBATCH --time=02:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate sn-torch-env

echo "Starting filtering and negative generation..."

# Set input/output file paths and score threshold
INPUT_FILE="/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/9606.protein_chemical.links.v5.0.tsv"
OUTPUT_FILE="/scratch/gpfs/jg9705/IW_data/Model_Resources/positives.tsv"
THRESHOLD=800

# Run the script
python filter_positives.py "$INPUT_FILE" "$OUTPUT_FILE" "$THRESHOLD"

echo "Filtering and negative generation completed!"
