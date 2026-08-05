#!/bin/bash
#SBATCH --job-name=hdf5_convert
#SBATCH --output=convert_%j.out
#SBATCH --error=convert_%j.err
#SBATCH --partition=cpu_p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH -A lbf@cpu

module purge

module load pytorch-gpu/py3/2.1.1

if [ -f ~/env_torch/bin/activate_pytorch ]; then
    echo "Activation de l'environnement utilisateur..."
    source ~/env_torch/bin/activate_pytorch
fi

pip install h5py tqdm --no-cache-dir

cd $SLURM_SUBMIT_DIR

H5_FILE="/lustre/fsn1/projects/rech/lbf/umn29tg/TDHF_combined.h5"

if [ -f "$H5_FILE" ]; then
    rm "$H5_FILE"
fi
lfs setstripe -c 8 "$H5_FILE"

echo "Début de la conversion..."
python convert_script.py
echo "Fin de la conversion."