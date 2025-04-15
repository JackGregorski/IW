#!/bin/bash

#SBATCH --job-name=KFold_ToyClassifier
#SBATCH --output=/scratch/gpfs/jg9705/IW_logs/%x.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_logs/%x.err
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-short
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate sn-torch-env

echo "Starting K-Fold training on toy datasets..."

# Define paths to toy dataset and embeddings
TRAIN_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/splits/train_toy.tsv"
TEST_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/splits/test_toy.tsv"
CHEM_FP_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/molecular_fingerprints.tsv"
PROT_EMB_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/encodings.tsv"
OUT_DIR="/scratch/gpfs/jg9705/IW_code/results_kfold_toy"

# Run your classifier script
python kfold-classifier.py \
    --train "${TRAIN_FILE}" \
    --test "${TEST_FILE}" \
    --chem_fp "${CHEM_FP_FILE}" \
    --prot_emb "${PROT_EMB_FILE}" \
    --out_dir "${OUT_DIR}"

echo "✅ K-Fold toy classifier run complete."
