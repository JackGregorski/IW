#!/bin/bash

#SBATCH --job-name=Filter_Chemicals
#SBATCH --output=/scratch/gpfs/jg9705/IW_data/logs/filter_chemicals.out
#SBATCH --error=/scratch/gpfs/jg9705/IW_data/logs/filter_chemicals.err
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate sn-torch-env

echo "Starting chemical filtering process..."

# Run the filtering script
python filter_chem_data.py

echo "Filtering completed successfully!"
