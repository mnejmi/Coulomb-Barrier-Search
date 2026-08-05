import os
import glob
import torch
import numpy as np
from tqdm import tqdm
TRAJ_DIR = '/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/TDHF90'
OUTPUT_DIR = '/lustre/fswork/projects/rech/lbf/umn29tg/UNO_TDHF_Project/output/particle_numbers'
os.makedirs(OUTPUT_DIR, exist_ok=True)
JOB_INDEX = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
N_JOBS = int(os.environ.get('N_JOBS', 10))

def load_trajectory(file_path):
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    if data.ndim == 4:
        rho = data[..., 0] + data[..., 1]
    else:
        rho = data
    return rho.numpy()
traj_files = sorted(glob.glob(os.path.join(TRAJ_DIR, 'data_2d_*.pt')))
traj_files = traj_files[:1383]
n_files = len(traj_files)
files_per_job = (n_files + N_JOBS - 1) // N_JOBS
start = JOB_INDEX * files_per_job
end = min(start + files_per_job, n_files)
traj_files_job = traj_files[start:end]
if len(traj_files_job) == 0:
    print(f'Job {JOB_INDEX}: no trajectories to process. Exiting.')
    exit(0)
print(f'Job {JOB_INDEX}: processing {len(traj_files_job)} trajectories ({start} to {end - 1})')
trajectory_dict = {}
for f in tqdm(traj_files_job, desc=f'Job {JOB_INDEX}'):
    rho = load_trajectory(f)
    N_frames = rho.sum(axis=(1, 2))
    N0 = N_frames[0]
    diff = N_frames - N0
    filename = os.path.basename(f)
    trajectory_dict[filename] = diff
output_file = os.path.join(OUTPUT_DIR, f'particle_numbers_job{JOB_INDEX}.npz')
np.savez(output_file, **trajectory_dict)
print(f' Saved dictionary results to {output_file}')