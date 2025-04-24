#!/bin/bash

#SBATCH --job-name=Eval_Models
#SBATCH --output=/scratch/gpfs/jg9705/IW_logs/%x.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_logs/%x.err
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=3
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code


module purge
module load anaconda3/2023.9
conda activate jg-torch-env

echo "Starting evaluation of all models on all test sets..."


python eval_model.py

echo "Model evaluation complete."
