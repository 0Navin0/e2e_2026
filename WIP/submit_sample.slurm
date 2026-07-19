#!/bin/bash
##SBATCH -q regular
#SBATCH --job-name=e2e_sample
#SBATCH --output=e2e_sample_%j.stdout
#SBATCH --time=1:00:00
#SBATCH --mem=200G
#SBATCH --partition=cosmology

###SBATCH --exclusive

# Activate your environment
source ~/.bashrc
conda activate dp2_smhm

python run_selection.py --env global_paths.yaml --config config.yaml --profile maglim_fluxlim_superset
