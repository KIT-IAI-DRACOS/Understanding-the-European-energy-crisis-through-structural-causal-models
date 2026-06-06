#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=50 
#SBATCH --mem=16G 
#SBATCH --time=02:00:00 
#SBATCH --job-name=run_model

module load devel/miniforge
conda activate causal_env
python scripts/03_gbt_training.py
