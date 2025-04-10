#!/bin/bash

#SBATCH --job-name=Enhanced_Interaction_Classifier
#SBATCH --output=/scratch/gpfs/jg9705/IW_logs/%x.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_logs/%x.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G                         # Increased for GPU support
#SBATCH --cpus-per-task=4                # Helps with DataLoader
#SBATCH --gres=gpu:1                     #  Request 1 GPU
#SBATCH --partition=gpu-short            # Or gpu-medium / gpu-long if needed
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate sn-torch-env

# Input paths
TRAIN_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/train.tsv"
VAL_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/val.tsv"
TEST_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/test.tsv"
CHEM_FP_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/molecular_fingerprints.tsv"
PROT_EMB_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/encodings.tsv"
OUT_DIR="/scratch/gpfs/jg9705/IW_code/results_enhanced"

echo "Starting enhanced training with:"
echo "  Train: ${TRAIN_FILE}"
echo "  Val:   ${VAL_FILE}"
echo "  Test:  ${TEST_FILE}"
echo "  Chem:  ${CHEM_FP_FILE}"
echo "  Prot:  ${PROT_EMB_FILE}"
echo "  Out:   ${OUT_DIR}"

python classifier_v2.py \
    --train "${TRAIN_FILE}" \
    --val "${VAL_FILE}" \
    --test "${TEST_FILE}" \
    --chem_fp "${CHEM_FP_FILE}" \
    --prot_emb "${PROT_EMB_FILE}" \
    --out_dir "${OUT_DIR}"

echo "Training and evaluation finished!"
