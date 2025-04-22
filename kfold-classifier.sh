#!/bin/bash
#SBATCH --job-name=KFold_Classifier
#SBATCH --output=/scratch/gpfs/jg9705/IW_logs/KFold_%A_%a.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_logs/KFold_%A_%a.err
#SBATCH --array=400,500,600,700,800,900      # Run one job per threshold
#SBATCH --time=03:00:00                       # Increase time per job
#SBATCH --mem=32G                             # Increase memory if needed
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate jg-torch-env

THRESHOLD=${SLURM_ARRAY_TASK_ID}
TRAIN_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/splits/threshold${THRESHOLD}/train.tsv"
CHEM_FP_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/molecular_fingerprints.tsv"
PROT_EMB_FILE="/scratch/gpfs/jg9705/IW_code/Model_Resources/encodings.tsv"
OUT_DIR="/scratch/gpfs/jg9705/IW_code/results_kfold/threshold${THRESHOLD}"

echo "Running threshold $THRESHOLD"

python kfold_classifier.py \
    --train "${TRAIN_FILE}" \
    --chem_fp "${CHEM_FP_FILE}" \
    --prot_emb "${PROT_EMB_FILE}" \
    --out_dir "${OUT_DIR}"

echo "✅ Done with threshold $THRESHOLD"
