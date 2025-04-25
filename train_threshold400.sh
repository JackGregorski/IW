#!/bin/bash
#SBATCH --job-name=Retrain400
#SBATCH --output=/scratch/gpfs/jg9705/IW_logs/retrain_400.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_logs/retrain_400.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jg9705@princeton.edu

module purge
module load anaconda3/2023.9
conda activate torch-env

python /scratch/gpfs/jg9705/IW_code/retrain_threshold400.py
