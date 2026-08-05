import os
import torch
import numpy as np
from tqdm import tqdm
input_dir = '/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/Trajectories/TDHF_1335_tensor'
output_dir = '/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/Trajectories/TDHF_1335_tensor_augmented'
os.makedirs(output_dir, exist_ok=True)

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

def save_array(path, arr):
    torch.save(arr, path)
for idx in tqdm(range(1, 1336)):
    fname = f'data_2d_{idx:04d}.pt'
    in_path = os.path.join(input_dir, fname)
    data = torch.load(in_path, weights_only=False)
    data = to_numpy(data)
    h_flip = np.flip(data, axis=2)
    v_flip = np.flip(data, axis=1)
    hv_flip = np.flip(np.flip(data, axis=1), axis=2)
    save_array(os.path.join(output_dir, f'data_2d_{idx:04d}.pt'), data)
    save_array(os.path.join(output_dir, f'data_2d_{idx + 1335:04d}.pt'), h_flip)
    save_array(os.path.join(output_dir, f'data_2d_{idx + 2 * 1335:04d}.pt'), v_flip)
    save_array(os.path.join(output_dir, f'data_2d_{idx + 3 * 1335:04d}.pt'), hv_flip)
print(' Augmented dataset created successfully!')