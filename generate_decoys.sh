#!/bin/bash

#SBATCH --job-name=Generate_Protein_Chem_Decoys
#SBATCH --output=/scratch/gpfs/jg9705/IW_data/logs/%x.out
#SBATCH --time=01:00:00
#SBATCH --mem=5G
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=jg9705@princeton.edu
#SBATCH -D /scratch/gpfs/jg9705/IW_code

module purge
module load anaconda3/2023.9
conda activate sn-torch-env

# Define script path
SCRIPT_PATH="/scratch/gpfs/jg9705/IW_code/generate_decoys.py"

echo "Running decoy generation script: $SCRIPT_PATH"

# Run the Python script
python "$SCRIPT_PATH"

echo "Decoy generation job completed!"
