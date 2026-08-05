import os
import torch
import sys
DATA_DIR = '/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/TDHF90'
N_TRAJ = 5532
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '-1'))
if task_id < 1 or task_id > N_TRAJ:
    print(f'Invalid SLURM_ARRAY_TASK_ID: {task_id}')
    sys.exit(1)
in_name = f'data_2d_{task_id:05d}.pt'
out_name = f'data_2d_{task_id + N_TRAJ:05d}.pt'
in_path = os.path.join(DATA_DIR, in_name)
out_path = os.path.join(DATA_DIR, out_name)
try:
    with torch.no_grad():
        if not os.path.exists(in_path):
            print(f' ERROR: Input file {in_path} not found')
            sys.exit(1)
        data = torch.load(in_path, map_location='cpu')
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        data = torch.flip(data, dims=(1,))
        data[..., 4] *= -1
        data[..., 5] *= -1
        torch.save(data, out_path)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        check_data = torch.load(out_path, map_location='cpu')
        if check_data.shape == data.shape:
            print(f' SUCCESS: Trajectory {task_id} → {task_id + N_TRAJ} verified.')
        else:
            print(f' ERROR: Shape mismatch for {out_name}!')
            sys.exit(1)
    else:
        print(f' ERROR: File {out_name} was not saved correctly!')
        sys.exit(1)
except Exception as e:
    print(f' CRITICAL ERROR for Task {task_id}: {e}')
    sys.exit(1)