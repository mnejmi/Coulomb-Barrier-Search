import os
import torch
import numpy as np
import random
from torch.utils.data import IterableDataset, get_worker_info
import torch.distributed as dist
import time

class UNOTIterableDataset(IterableDataset):

    def __init__(self, file_list, inputpath, predstep, iteration, shuffle=True):
        self.file_list = file_list
        self.inputpath = inputpath
        self.predstep = predstep
        self.iteration = iteration
        self.shuffle = shuffle

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        total_splits = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id
        local_files = [f for i, f in enumerate(self.file_list) if i % total_splits == global_worker_id]
        while True:
            if self.shuffle:
                random.shuffle(local_files)
            for file_name in local_files:
                full_path = os.path.join(self.inputpath, file_name)
                try:
                    if file_name.endswith('.npy'):
                        phis = np.load(full_path)
                    else:
                        phis = torch.load(full_path, map_location='cpu', weights_only=False)
                        if torch.is_tensor(phis):
                            phis = phis.numpy()
                except Exception as e:
                    print(f'[Worker {global_worker_id}] Failed to load {file_name}: {e}')
                    continue
                num_frames = phis.shape[0]
                max_start = num_frames - self.predstep * self.iteration
                if max_start <= 0:
                    continue
                frame_indices = list(range(max_start))
                if self.shuffle:
                    random.shuffle(frame_indices)
                for frame_idx in frame_indices:
                    a = phis[frame_idx]
                    u_frames = []
                    for i in range(1, self.iteration + 1):
                        u_frames.append(phis[frame_idx + i * self.predstep])
                    u = np.stack(u_frames, axis=0)
                    yield (torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(u, dtype=torch.float32))
                del phis

def data_prepare(cfg, inputpath):
    start_time = time.time()
    all_files = [f for f in os.listdir(inputpath) if f.startswith('data_2d_') and (f.endswith('.pt') or f.endswith('.npy'))]
    all_files.sort()
    if len(all_files) == 0:
        raise ValueError(f'No valid data files found in {inputpath}')
    ratio = getattr(cfg, 'val_ratio', 0.1)
    seed_number = getattr(cfg, 'trajectories_seed', 42)
    predstep = getattr(cfg, 'pred_step', 1)
    iteration = getattr(cfg, 'iteration', 4)
    import re
    from collections import defaultdict
    groups = defaultdict(list)
    for f in all_files:
        match = re.search('data_2d_(\\d+)\\.(pt|npy)', f)
        if match:
            idx = int(match.group(1))
            base_id = (idx - 1) % 1383
            groups[base_id].append(f)
    group_keys = list(groups.keys())
    group_keys.sort()
    random.seed(seed_number)
    random.shuffle(group_keys)
    limit_data = getattr(cfg, 'limit_data', None)
    if limit_data is not None and limit_data > 0:
        target_groups = max(1, int(limit_data / 8))
        group_keys = group_keys[:target_groups]
        print(f'️ DEBUG MODE: Limiting dataset to {target_groups} base symmetries (~{target_groups * 8} files).')
    Train_dim_groups = int(len(group_keys) * (1 - ratio))
    train_keys = group_keys[:Train_dim_groups]
    val_keys = group_keys[Train_dim_groups:]
    Train_list = []
    for k in train_keys:
        Train_list.extend(groups[k])
    Val_list = []
    for k in val_keys:
        Val_list.extend(groups[k])
    random.shuffle(Train_list)
    random.shuffle(Val_list)
    limit_data = getattr(cfg, 'limit_data', None)
    if limit_data is not None:
        Train_list = Train_list[:int(limit_data * (1 - ratio))]
        Val_list = Val_list[:int(limit_data * ratio)]
    print(f'→ Preparing lazily evaluated IterableDataset: {len(Train_list)} training trajectories, {len(Val_list)} validation trajectories')
    train_dataset = UNOTIterableDataset(Train_list, inputpath, predstep, iteration, shuffle=True)
    val_dataset = UNOTIterableDataset(Val_list, inputpath, predstep, iteration, shuffle=False)
    print(f' Dataset configuration complete in {time.time() - start_time:.2f} seconds')
    return (train_dataset, val_dataset)