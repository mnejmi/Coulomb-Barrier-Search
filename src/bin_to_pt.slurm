#!/bin/bash
#SBATCH --job-name=bin2pt
#SBATCH --output=logs/bin2pt_%A_%a.out
#SBATCH --error=logs/bin2pt_%A_%a.err
#SBATCH --array=0-60
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_p1
#SBATCH -A lbf@cpu
#SBATCH --hint=nomultithread

set -euo pipefail

mkdir -p logs

module purge
module load pytorch-gpu/py3/2.8.0

python bin_to_pt.py