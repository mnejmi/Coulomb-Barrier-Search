#!/bin/bash
#SBATCH --job-name=particle_numbers
#SBATCH --output=logs/particle_numbers_%A_%a.out
#SBATCH --error=logs/particle_numbers_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_p1
#SBATCH --account=lbf@cpu
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=8

module purge
module load pytorch-gpu/py3/2.8.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_JOBS=10

echo "=========================================="
echo "Starting Task ID   : $SLURM_ARRAY_TASK_ID"
echo "Running on Node    : $(hostname)"
echo "Allocated CPU cores: $SLURM_CPUS_PER_TASK"
echo "=========================================="

srun python compute_particle_numbers_cpu.py