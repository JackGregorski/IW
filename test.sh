#!/bin/bash
#SBATCH --job-name=test_rh9_gpu
#SBATCH --output=test_rh9_gpu.out
#SBATCH --error=test_rh9_gpu.err
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --constraint=rh9

module purge
module load anaconda3/2023.9
conda activate jg-torch-env

python -c "import torch; print('✅ Torch version:', torch.__version__)"
