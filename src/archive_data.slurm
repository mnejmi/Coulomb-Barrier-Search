#!/bin/bash
#SBATCH --job-name=tar_data
#SBATCH --output=tar_data_%j.out
#SBATCH --error=tar_data_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH -A umn29tg

cd /lustre/fsn1/projects/rech/lbf/umn29tg/TDHF_rot90

mkdir -p $STORE/DATA/Archive_TDHF/

echo "Début de l'archivage le : $(date)"

tar -czf $STORE/DATA/Archive_TDHF/trajectoires_backup.tar ./*.pt

echo "Fin de l'archivage le : $(date)"