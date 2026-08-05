#!/bin/bash
#SBATCH --job-name=TDHF_stats
#SBATCH --output=logs/stats_%j.out
#SBATCH --error=logs/stats_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --partition=cpu_p1
#SBATCH --account=lbf@cpu
#SBATCH --hint=nomultithread

set -euo pipefail

echo "Starting job..."
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

module purge
module load pytorch-gpu/py3/2.8.0     

echo "Running statistics computation..."

srun --cpu-bind=cores python3 -u cal_m_std.py

echo "Job finished successfully!"