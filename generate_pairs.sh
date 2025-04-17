#!/bin/bash

#SBATCH --job-name=Filter_PosNeg
#SBATCH --output=/scratch/gpfs/jg9705/IW_logs/filter_pos_neg_%j.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_logs/filter_pos_neg_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate jg-torch-env

echo "Starting filtering and negative generation..."

# Set input file and output base path (no extension)
INPUT_FILE="/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/9606.protein_chemical.links.v5.0.tsv"
OUTPUT_BASE="/scratch/gpfs/jg9705/IW_data/Model_Resources/pairs"
THRESHOLDS="400,500,600,700,800,900"

# Run the updated script
python filter_positives.py "$INPUT_FILE" "$OUTPUT_BASE" "$THRESHOLDS"

echo "Filtering and negative generation completed!"
