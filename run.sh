#!/bin/bash
#SBATCH -t 00:10:00
#SBATCH -J reachCube
#SBATCH -o output_%j.log
#SBATCH -e error_%j.err

module load Anaconda3
source config_conda.sh
conda activate safe_bimanual_rl

# Aller dans le bon dossier
cd ~/safe-bimanual-rl

# Lancer l'expérience
python3 -m safe_bimanual_rl.reach_point_experiment_sac