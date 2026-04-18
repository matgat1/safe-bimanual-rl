#!/bin/bash

# Parameters
#SBATCH --array=0-2%3
#SBATCH --error=/home/matgat/safe-bimanual-rl/multirun/2026-04-15/14-53-05/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --job-name=reachCube
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/matgat/safe-bimanual-rl/multirun/2026-04-15/14-53-05/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --signal=USR2@90
#SBATCH --time=120
#SBATCH --wckey=submitit

# setup
module load Anaconda3
source config_conda.sh
conda activate safe_bimanual_rl

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/matgat/safe-bimanual-rl/multirun/2026-04-15/14-53-05/.submitit/%A_%a/%A_%a_%t_log.out --error /home/matgat/safe-bimanual-rl/multirun/2026-04-15/14-53-05/.submitit/%A_%a/%A_%a_%t_log.err /home/matgat/.conda/envs/safe_bimanual_rl/bin/python -u -m submitit.core._submit /home/matgat/safe-bimanual-rl/multirun/2026-04-15/14-53-05/.submitit/%j
