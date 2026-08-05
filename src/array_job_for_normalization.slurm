#!/bin/bash
#SBATCH --job-name=TDHF_norm_array
#SBATCH --output=logs/norm_%A_%a.out
#SBATCH --error=logs/norm_%A_%a.err
#SBATCH --array=0-49%10
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_p1
#SBATCH --account=lbf@cpu
#SBATCH --hint=nomultithread

set -euo pipefail

mkdir -p logs

echo "Task ID: $SLURM_ARRAY_TASK_ID"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

module purge
module load pytorch-gpu/py3/2.8.0

srun python3 -u preprocess_tdhf_parallel.py