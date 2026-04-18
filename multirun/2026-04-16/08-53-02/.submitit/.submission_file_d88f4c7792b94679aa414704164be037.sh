#!/bin/bash

# Parameters
#SBATCH --error=/home/matgat/safe-bimanual-rl/multirun/2026-04-16/08-53-02/.submitit/%j/%j_0_log.err
#SBATCH --job-name=reachCube
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/matgat/safe-bimanual-rl/multirun/2026-04-16/08-53-02/.submitit/%j/%j_0_log.out
#SBATCH --signal=USR2@90
#SBATCH --time=120
#SBATCH --wckey=submitit

# setup
module load Anaconda3
source config_conda.sh
conda activate safe_bimanual_rl

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/matgat/safe-bimanual-rl/multirun/2026-04-16/08-53-02/.submitit/%j/%j_%t_log.out --error /home/matgat/safe-bimanual-rl/multirun/2026-04-16/08-53-02/.submitit/%j/%j_%t_log.err /home/matgat/.conda/envs/safe_bimanual_rl/bin/python -u -m submitit.core._submit /home/matgat/safe-bimanual-rl/multirun/2026-04-16/08-53-02/.submitit/%j
