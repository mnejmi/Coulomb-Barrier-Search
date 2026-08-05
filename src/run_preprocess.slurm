#!/bin/bash
#SBATCH --job-name=tdhf_norm
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=norm_%j.log
#SBATCH --partition=cpu_p1
#SBATCH -A lbf@cpu
#SBATCH --hint=nomultithread

module purge
source ~/env_torch/bin/activate_pytorch

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun python preprocess_tdhf_parallel.py --n_workers $SLURM_CPUS_PER_TASK