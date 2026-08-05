#!/bin/bash
#SBATCH --job-name=norm_stats
#SBATCH --output=norm_stats_%j.out
#SBATCH --error=norm_stats_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH -A lbf@cpu

echo "=============================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Start     : $(date)"
echo "=============================="

module purge
source ~/env_torch/bin/activate_pytorch

cd /lustre/fsn1/projects/rech/lbf/umn29tg/TRAIN/old3/src || exit 1

export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

python compute_norm_stats.py

echo "=============================="
echo "End : $(date)"
echo "=============================="