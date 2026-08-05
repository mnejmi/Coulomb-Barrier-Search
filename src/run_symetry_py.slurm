#!/bin/bash
#SBATCH --job-name=TDHF_x
#SBATCH --output=logs/x_%A_%a.out
#SBATCH --error=logs/x_%A_%a.err
#SBATCH --array=1-5532%200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --partition=cpu_p1
#SBATCH -A lbf@cpu
#SBATCH --hint=nomultithread

set -euo pipefail

mkdir -p logs

module purge
module load pytorch-gpu/py3/2.8.0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python augment_tdhf_i.py