#!/bin/bash
#SBATCH --job-name=plot_gpu
#SBATCH --output=logs/plot_%j.out
#SBATCH --error=logs/plot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=cpu_p1
#SBATCH --account=lbf@cpu

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p logs

module purge
module load python
module load gnuplot/5.4.3

CSV_FILE="/lustre/fsn1/projects/rech/lbf/umn29tg/TRAIN/UNO_1GPU/OutPuts/1Nepoc202_lr5em4_predstep_iter1_nselect11064_bz128_width20_loadmodel0_0/logs/gpu_stats_1153993.csv"

echo "Cleaning CSV..."
python clean_gpu_csv.py $CSV_FILE

CLEAN_FILE="${CSV_FILE%.csv}_clean.csv"

if [ ! -f "$CLEAN_FILE" ]; then
    echo " ERROR: Clean file not found!"
    exit 1
fi

echo "Plotting..."
gnuplot -c plot_gpu.gp "$CLEAN_FILE"

echo " Done!"