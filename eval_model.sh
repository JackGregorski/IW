#!/bin/bash

#SBATCH --job-name=Eval_Models
#SBATCH --output=/scratch/gpfs/jg9705/IW_logs/%x.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_logs/%x.err
#SBATCH --time=01:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=3
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

# Load appropriate environment
module purge
module load anaconda3/2023.9
conda activate jg-torch-env

echo "Starting evaluation of all models on all test sets..."

# Run your evaluation script (assume it's called evaluate_models.py or similar)
python eval_model.py

echo "✅ Model evaluation complete."
