#!/bin/bash

#SBATCH --job-name=Generate_Protein_Encodings
#SBATCH --output=/scratch/gpfs/jg9705/IW_data/gen_protein_encodings/logs/%x.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_data/gen_protein_encodings/logs/%x.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate jg-torch-env

# Define the input FASTA file (contains all ~170K proteins)
INPUT_FASTA="/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/9606.protein.sequences.v12.0.fa"

# File containing the list of 19,196 proteins to encode
UNIQUE_PROTEINS="/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/unique_proteins.tsv"

# Path to the locally downloaded ESM model
MODEL_FILE="/scratch/gpfs/jg9705/IW_data/gen_protein_encodings/esm1v_t33_650M_UR90S_1.pt"

# Output location for protein encodings
OUTPUT_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/encodings.tsv"

echo "Starting protein encoding with model: ${MODEL_FILE}"
echo "Processing only proteins listed in: ${UNIQUE_PROTEINS}"

python gen_encodings.py "${INPUT_FASTA}" "${UNIQUE_PROTEINS}" "${OUTPUT_FILE}" "${MODEL_FILE}"

echo "✅ Job completed successfully!"
