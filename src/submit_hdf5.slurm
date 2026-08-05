#!/bin/bash
#SBATCH --job-name=TDHF_h5
#SBATCH --output=logs/h5_%j.out
#SBATCH --error=logs/h5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --partition=cpu_p1
#SBATCH --account=lbf@cpu
#SBATCH --hint=nomultithread

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

module purge
module load pytorch-gpu/py3/2.8.0     

echo "Starting HDF5 conversion using 8 threads..."

srun python3 -u convert_script.py

echo "Job finished successfully!"