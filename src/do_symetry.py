import os
import torch
from tqdm import tqdm
input_dir = '/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/TDHF'
output_dir = '/lustre/fsstor/projects/rech/lbf/umn29tg/DATA/TDHF90'
os.makedirs(output_dir, exist_ok=True)
for fname in tqdm(sorted(os.listdir(input_dir))):
    if not fname.endswith('.pt'):
        continue
    in_path = os.path.join(input_dir, fname)
    out_path = os.path.join(output_dir, fname)
    data = torch.load(in_path, weights_only=False)
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    data_rot = torch.rot90(data, k=1, dims=(1, 2))
    torch.save(data_rot, out_path)
print(' All trajectories rotated by +90° and saved.')