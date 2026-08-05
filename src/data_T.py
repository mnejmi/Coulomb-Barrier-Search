import os
import torch
from tqdm import tqdm
data_dir = '/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/TDHF_rot90'
n_traj = 1383
for i in tqdm(range(1, n_traj + 1), desc='Processing TDHF trajectories'):
    in_name = f'data_2d_{i:04d}.pt'
    out_name = f'data_2d_{i + n_traj:04d}.pt'
    in_path = os.path.join(data_dir, in_name)
    out_path = os.path.join(data_dir, out_name)
    data = torch.load(in_path, weights_only=False)
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    data = data.transpose(1, 2)
    data[..., 2:] *= -1
    data = data[..., [0, 1, 4, 5, 2, 3]]
    torch.save(data, out_path)
print(' All trajectories processed successfully!')