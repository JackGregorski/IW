#!/bin/bash

#SBATCH --job-name=Generate_Chemical_Embeddings
#SBATCH --output=/scratch/gpfs/jg9705/IW_data/Generate_Chemical_Embeddings/logs/%x.out
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate sn-torch-env

# Define input and output file paths
INPUT_FILE="/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/chemicals.v5.0.tsv"
OUTPUT_FILE="/scratch/gpfs/jg9704/IW_data/Generate_Chemical_Embeddings/molecular_fingerprints.txt"

echo "Processing SMILES from: $INPUT_FILE"
echo "Saving embeddings to: $OUTPUT_FILE"

# Run the Python script
python gen_mol_fingerprint.py "$INPUT_FILE" "$OUTPUT_FILE"

echo "Job completed successfully!"
