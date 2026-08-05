import numpy as np
import torch
import os
import sys
INPUT_DIR = '/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/Bin_TDHF/'
OUTPUT_DIR = '/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/Trajectories/New/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
bin_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.bin')])
if task_id >= len(bin_files):
    print(f'Task {task_id} out of range.')
    sys.exit(0)
bin_name = bin_files[task_id]
full_path = os.path.join(INPUT_DIR, bin_name)
print(f'[Task {task_id}] Processing: {bin_name}')
phis_list = []
with open(full_path, 'rb') as f:
    while True:
        try:
            nxyz = np.fromfile(f, dtype=np.intc, count=3)
            if nxyz.size < 3:
                break
            dx = np.fromfile(f, dtype=np.double, count=1)
            if dx.size < 1:
                break
            time = np.fromfile(f, dtype=np.double, count=1)
            if time.size < 1:
                break
            nx, ny, nz = nxyz
            count = nx * nz * 6
            matrix = np.fromfile(f, dtype=np.float32, count=count)
            if matrix.size < count:
                break
            phist = matrix.reshape(nx, nz, 6)
            phis_list.append(phist.copy())
        except Exception as e:
            print(f'Stopped at task {task_id} due to: {e}')
            break
if not phis_list:
    print('No data found in file.')
    sys.exit(1)
phis = np.stack(phis_list, axis=0)
phis = torch.from_numpy(phis)
phis = torch.rot90(phis, k=1, dims=(1, 2))
print(f'Final Shape: {phis.shape}')
out_name = bin_name.replace('.bin', '.pt')
out_path = os.path.join(OUTPUT_DIR, out_name)
torch.save(phis, out_path)
print(f'[Task {task_id}] Saved: {out_path}')