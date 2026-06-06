#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=96
#SBATCH --mem=64G 
#SBATCH --time=04:00:00 
#SBATCH --job-name=run_flow_FR_price

module load devel/miniforge
conda activate causal_env
python scripts/04_gbt_shapley_flow.py --target "FR_price"
