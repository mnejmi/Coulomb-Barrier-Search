import os
import torch
from glob import glob
DATA_DIR = '/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/TDHF90'
OUTPUT_DIR = '/lustre/fsn1/projects/rech/lbf/umn29tg/TDHF_normalized'
STATS_PATH = '/lustre/fswork/projects/rech/lbf/umn29tg/UNO_TDHF_Project/output/dataset_stats.pt'
N_JOBS = 50
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', -1))
if task_id < 0:
    raise RuntimeError('Not running inside SLURM array')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)
stats = torch.load(STATS_PATH, map_location='cpu')
MEAN_S = stats['mean'].view(1, 1, 1, -1)
STD_S = torch.clamp(stats['std'].view(1, 1, 1, -1), min=1e-12)
files = sorted(glob(os.path.join(DATA_DIR, 'data_2d_*.pt')))
n_files = len(files)
chunk_size = (n_files + N_JOBS - 1) // N_JOBS
start = task_id * chunk_size
end = min(start + chunk_size, n_files)
if start >= n_files:
    print(f'[Task {task_id}] No files to process (start index {start} >= total files {n_files}). Exiting.')
    exit(0)
subset = files[start:end]
print(f'[Task {task_id}] Processing {len(subset)} files ({start}:{end})')
os.makedirs(OUTPUT_DIR, exist_ok=True)
processed = 0
skipped = 0
for filepath in subset:
    try:
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(filepath))
        if os.path.exists(out_path):
            skipped += 1
            continue
        traj = torch.load(filepath, map_location='cpu').float()
        traj = (traj - MEAN_S) / STD_S
        torch.save(traj, out_path, _use_new_zipfile_serialization=False)
        processed += 1
    except Exception as e:
        print(f'Error: {filepath} → {e}')
print(f'[Task {task_id}] Done {processed}/{len(subset)} | skipped={skipped}')