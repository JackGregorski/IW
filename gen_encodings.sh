#!/bin/bash
#SBATCH --job-name=Generate_Protein_Encodings
#SBATCH --array=0-1  # Adjust array size as needed
#SBATCH --output=/scratch/gpfs/jg9705/IW_data/gen_protein_encodings/encodings/%x-%a.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_data/gen_protein_encodings/logs/%x-%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate sn-torch-env

# Define the input FASTA file path
INPUT_FASTA="scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/9606.protein.sequences.v12.0.fa"

# Define the model file path
MODEL_FILE="scratch/gpfs/jg9705/IW_data/gen_protein_encodings/esm1v_t33_650M_UR90S_1.pt"

# Define the output file (include the array index in the filename)
OUTPUT_FILE="scratch/gpfs/jg9705/IW_data/gen_protein_encodings/encodings_${SLURM_ARRAY_TASK_ID}.tsv"

echo "Running task ${SLURM_ARRAY_TASK_ID}: processing FASTA file ${INPUT_FASTA}"
python gen_encodings.py "${INPUT_FASTA}" "${OUTPUT_FILE}" "${MODEL_FILE}"
