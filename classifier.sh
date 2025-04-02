#!/bin/bash

#SBATCH --job-name=Train_Interaction_Classifier
#SBATCH --output=/scratch/gpfs/jg9705/IW_logs/%x.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_logs/%x.err
#SBATCH --time=04:00:00            # Increased time for Optuna trials
#SBATCH --mem=24G                  # Optional: increase for safety during tuning
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate sn-torch-env

# Define input file paths
TRAIN_FILE="/scratch/gpfs/jg9705/IW_data/Model_Resources/train.tsv"
VAL_FILE="/scratch/gpfs/jg9705/IW_data/Model_Resources/val.tsv"
TEST_FILE="/scratch/gpfs/jg9705/IW_data/Model_Resources/test.tsv"
CHEM_FP_FILE="/scratch/gpfs/jg9705/IW_data/Model_Resources/molecular_fingerprints.tsv"
PROT_EMB_FILE="/scratch/gpfs/jg9705/IW_data/Model_Resources/encodings.tsv"
OUT_DIR="/scratch/gpfs/jg9705/IW_code/"

echo "Starting training with:"
echo "  Train set: ${TRAIN_FILE}"
echo "  Val set:   ${VAL_FILE}"
echo "  Test set:  ${TEST_FILE}"
echo "  Chemical fingerprints: ${CHEM_FP_FILE}"
echo "  Protein embeddings:    ${PROT_EMB_FILE}"
echo "  Output directory:      ${OUT_DIR}"

python classifier.py \
    --train "${TRAIN_FILE}" \
    --val "${VAL_FILE}" \
    --test "${TEST_FILE}" \
    --chem_fp "${CHEM_FP_FILE}" \
    --prot_emb "${PROT_EMB_FILE}" \
    --out_dir "${OUT_DIR}"

echo "Training and evaluation completed successfully!"
