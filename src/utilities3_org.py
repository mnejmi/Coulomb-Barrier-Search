import torch
import numpy as np
import scipy.io
import torch.nn as nn
import os
import h5py
import operator
from functools import reduce
from functools import partial
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, get_worker_info
from glob import glob
import random
import gc
import h5py
import time
import yaml
import signal
import sys
import inspect
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor

class SlurmTimeout(Exception):
    pass

def handle_slurm_timeout(signum, frame):
    print('\n[WARNING] SLURM 10-minute timeout signal received!', flush=True)
    raise SlurmTimeout('Time to save and exit!')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from typing import List

@dataclass
class TrainConfig:
    num_nodes: int
    n_gpu_per_node: int
    multi_gpu: bool
    use_gpu: bool
    seed: int
    dx: float
    dy: float
    dt: float
    input_shape: List[int]
    num_epochs: int
    batch_size: int
    iteration: int
    pred_step: int
    predict_residual: bool
    base_lr: float
    min_lr: float
    weight_decay: float
    scheduler_step: int
    scheduler_gamma: float
    width: int
    uno_layers: int
    use_constraint: bool
    alpha: float
    beta: float
    lp_p: int
    val_ratio: float
    test_ratio: float
    n_select: int
    test_num: int
    project_tag: str
    save_model: bool
    save_frequency: int
    checkpoint_dir: str
    load_model: bool
    resume_path: str

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    flat_config = {}
    for section in config_dict.values():
        if isinstance(section, dict):
            flat_config.update(section)
    sig = inspect.signature(TrainConfig)
    valid_keys = sig.parameters.keys()
    filtered_config = {k: v for k, v in flat_config.items() if k in valid_keys}
    print('\n' + '=' * 60)
    print(f' INITIALIZING EXPERIMENT: {filtered_config.get('project_tag', 'Unknown')}')
    print('=' * 60)
    for key in sorted(filtered_config.keys()):
        value = filtered_config[key]
        print(f'{key:.<30} {value}')
    ignored = set(flat_config.keys()) - set(filtered_config.keys())
    if ignored:
        print('-' * 60)
        print(f'️  WARNING: The following YAML keys were ignored (not in TrainConfig):')
        print(f'   {', '.join(ignored)}')
    print('=' * 60 + '\n')
    return TrainConfig(**filtered_config)

def load_split_to_vram(data_dir, cfg, split, device, rank=0, world_size=1):
    print(f'[Rank {rank} | {split.upper()}] Scanning directory {data_dir}...')
    dt_index = cfg.pred_step * cfg.iteration
    all_files = sorted(glob(os.path.join(data_dir, 'data_2d_*.pt')))
    if len(all_files) == 0:
        raise ValueError(f' ERROR: No files found in {data_dir}')
    all_file_names = [os.path.basename(f) for f in all_files]
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(all_file_names)
    total = len(all_file_names)
    n_val = int(total * cfg.val_ratio)
    n_test = int(total * cfg.test_ratio)
    n_train = total - n_val - n_test
    if split == 'train':
        split_names = all_file_names[:n_train]
    elif split == 'val':
        split_names = all_file_names[n_train:n_train + n_val]
    elif split == 'test':
        split_names = all_file_names[n_train + n_val:]
    else:
        raise ValueError("split must be 'train', 'val', or 'test'")
    if split in ['train', 'val']:
        total_split = len(split_names)
        chunk_size = total_split // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank != world_size - 1 else total_split
        split_names = split_names[start_idx:end_idx]
    print(f'[Rank {rank} | {split.upper()}] Loading its chunk of {len(split_names)} files...')
    NCPU = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))

    def get_shape(fname):
        f_path = os.path.join(data_dir, fname)
        ds_shape = torch.load(f_path, map_location='cpu', weights_only=True).shape[0]
        return (fname, ds_shape)
    with ThreadPoolExecutor(max_workers=NCPU) as executor:
        shapes = list(executor.map(get_shape, split_names))
    file_tasks = []
    current_idx = 0
    for fname, ds_shape in shapes:
        if ds_shape > dt_index:
            valid_steps = ds_shape - dt_index
            file_tasks.append((fname, current_idx, valid_steps))
            current_idx += valid_steps
    total_samples = current_idx
    X_all = torch.empty((total_samples, 56, 56, 6), dtype=torch.float32)
    Y_all = torch.empty((total_samples, 56, 56, 6), dtype=torch.float32)
    DN_all = torch.empty((total_samples,), dtype=torch.float32)

    def process_file(task):
        fname, start_idx, valid_steps = task
        f_path = os.path.join(data_dir, fname)
        data = torch.load(f_path, map_location='cpu', weights_only=True)
        frames_x = data[:valid_steps]
        frames_future = data[dt_index:dt_index + valid_steps]
        if cfg.predict_residual == 1:
            frames_y = frames_future - frames_x
            density_diff = frames_y[:, :, :, 0] + frames_y[:, :, :, 1]
            delta_N = torch.sum(density_diff, dim=(1, 2)) * cfg.dx * cfg.dy
        else:
            frames_y = frames_future
            density_y = frames_y[:, :, :, 0] + frames_y[:, :, :, 1]
            N_y = torch.sum(density_y, dim=(1, 2)) * cfg.dx * cfg.dy
            density_x = frames_x[:, :, :, 0] + frames_x[:, :, :, 1]
            N_x = torch.sum(density_x, dim=(1, 2)) * cfg.dx * cfg.dy
            delta_N = N_y - N_x
        X_all[start_idx:start_idx + valid_steps] = frames_x
        Y_all[start_idx:start_idx + valid_steps] = frames_y
        DN_all[start_idx:start_idx + valid_steps] = delta_N
        del data
    with ThreadPoolExecutor(max_workers=NCPU) as executor:
        list(executor.map(process_file, file_tasks))
    print(f'[Rank {rank} | {split.upper()}] Successfully loaded {total_samples} samples into System RAM.')
    return (X_all, Y_all, DN_all, split_names)

class TrajectoryDataset(Dataset):

    def __init__(self, h5_path, cfg: TrainConfig, split='train'):
        self.h5_path = h5_path
        self.index = cfg.pred_step * cfg.iteration
        self.predict_residual = cfg.predict_residual
        self.samples = []
        self.dx = cfg.dx
        self.dy = cfg.dy
        with h5py.File(h5_path, 'r') as f:
            all_traj_keys = sorted(list(f.keys()))
            rng = np.random.default_rng(cfg.seed)
            rng.shuffle(all_traj_keys)
            if cfg.n_select is not None and cfg.n_select < len(all_traj_keys):
                print(f'[INFO] Selecting {cfg.n_select} random trajectories out of {len(all_traj_keys)}')
                all_traj_keys = all_traj_keys[:cfg.n_select]
            total = len(all_traj_keys)
            n_val = int(total * cfg.val_ratio)
            n_test = int(total * cfg.test_ratio)
            n_train = total - n_val - n_test
            total = len(all_traj_keys)
            n_val = int(total * cfg.val_ratio)
            n_test = int(total * cfg.test_ratio)
            n_train = total - n_val - n_test
            if split == 'train':
                self.keys = all_traj_keys[:n_train]
            elif split == 'val':
                self.keys = all_traj_keys[n_train:n_train + n_val]
            elif split == 'test':
                self.keys = all_traj_keys[n_train + n_val:]
            else:
                raise ValueError("split must be 'train', 'val', or 'test'")
            print(f'[{split.upper()}] Assigned {len(self.keys)} trajectories.')
            for key in self.keys:
                ds_shape = f[key].shape[0]
                if ds_shape > cfg.pred_step * cfg.iteration:
                    for t in range(ds_shape - cfg.pred_step * cfg.iteration):
                        self.samples.append((key, t))
        self.file = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r', swmr=True)
        ds_name, t_idx = self.samples[idx]
        data = self.file[ds_name]
        if self.predict_residual == 1:
            frame_x = data[t_idx]
            frame_y = data[t_idx + self.index] - data[t_idx]
        else:
            frame_x = data[t_idx]
            frame_y = data[t_idx + self.index]
        density_x = frame_x[:, :, 0] + frame_x[:, :, 1]
        N_x = np.sum(density_x) * self.dx * self.dy
        if self.predict_residual == 1:
            density_diff = frame_y[:, :, 0] + frame_y[:, :, 1]
            delta_N = np.sum(density_diff) * self.dx * self.dy
        else:
            density_y = frame_y[:, :, 0] + frame_y[:, :, 1]
            N_y = np.sum(density_y) * self.dx * self.dy
            delta_N = N_y - N_x
        x_tensor = torch.from_numpy(frame_x).float()
        y_tensor = torch.from_numpy(frame_y).float()
        if self.predict_residual == 1:
            delta_N_tensor = torch.tensor(delta_N, dtype=torch.float32)
            return (x_tensor, y_tensor, delta_N_tensor)
        else:
            return (x_tensor, y_tensor, delta_N)

def split_files(data_dir, cfg, seed=42):
    files = sorted(glob(os.path.join(data_dir, 'data_2d_*.pt')))
    random.seed(seed)
    random.shuffle(files)
    selected_files = files[:cfg.n_select]
    n = len(selected_files)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    train_files = selected_files[:n_train]
    val_files = selected_files[n_train:n_train + n_val]
    test_files = selected_files[n_train + n_val:]
    print('================================')
    print(f'Total available trajectories : {len(files)}')
    print(f'Selected trajectories        : {len(selected_files)}')
    print(f'Train : {len(train_files)}')
    print(f'Val   : {len(val_files)}')
    print(f'Test  : {len(test_files)}')
    print('================================')
    return (train_files, val_files, test_files)

class HDF5TDHFDataset(Dataset):

    def __init__(self, h5_path, stats_path, pred_step=4, split_keys=None):
        self.h5_path = h5_path
        self.pred_step = pred_step
        self.keys = split_keys
        stats = torch.load(stats_path, map_location='cpu')
        self.mean = stats['mean'].view(1, 1, 6)
        self.std = stats['std'].view(1, 1, 6)
        self.index_map = []
        self.h5_file = None
        with h5py.File(self.h5_path, 'r') as f:
            if self.keys is None:
                self.keys = sorted(list(f.keys()))
            for key in self.keys:
                T = f[key].shape[0]
                for t in range(T - pred_step):
                    self.index_map.append((key, t))
        print(f'Dataset initialisé. {len(self.index_map)} échantillons prêts.')

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        key, t = self.index_map[idx]
        dataset = self.h5_file[key]
        x_np = dataset[t]
        y_np = dataset[t + self.pred_step]
        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()
        return (x, y)

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

def get_split_keys(h5_path, cfg, seed=42):
    with h5py.File(h5_path, 'r') as f:
        keys = sorted(list(f.keys()))
    random.seed(seed)
    random.shuffle(keys)
    selected_keys = keys[:cfg.n_select]
    n = len(selected_keys)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    train_keys = selected_keys[:n_train]
    val_keys = selected_keys[n_train:n_train + n_val]
    test_keys = selected_keys[n_train + n_val:]
    return (train_keys, val_keys, test_keys)

class TDHFFrameDataset(Dataset):

    def __init__(self, data_dir, stats_path, pred_step=4, max_frames=None):
        if data_dir:
            self.files = sorted(glob(os.path.join(data_dir, 'data_2d_*.pt')))
        else:
            self.files = []
        self.pred_step = pred_step
        self.max_frames = max_frames
        stats = torch.load(stats_path, map_location='cpu')
        self.mean = stats['mean'].view(1, 1, 1, 6)
        self.std = stats['std'].view(1, 1, 1, 6)
        self.index_map = []
        for f in self.files:
            try:
                traj = torch.load(f, map_location='cpu')
                T = traj.shape[0]
                if self.max_frames:
                    T = min(T, self.max_frames)
                for t in range(T - pred_step):
                    self.index_map.append((f, t))
            except Exception as e:
                print('Skipping file:', f, e)
        print('Total samples in dataset:', len(self.index_map))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        f, t = self.index_map[idx]
        traj = torch.load(f, map_location='cpu').float()
        x = traj[t]
        y = traj[t + self.pred_step]
        x = (x - self.mean.view(1, 1, 6)) / self.std.view(1, 1, 6)
        y = (y - self.mean.view(1, 1, 6)) / self.std.view(1, 1, 6)
        return (x, y)

class TDHFFrameDatasetFromList(TDHFFrameDataset):

    def __init__(self, file_list, stats_path, pred_step=3):
        super().__init__(data_dir='', stats_path=stats_path, pred_step=pred_step)
        self.files = file_list
        self.index_map = []
        for f in self.files:
            traj = torch.load(f, map_location='cpu')
            T = traj.shape[0]
            for t in range(T - self.pred_step):
                self.index_map.append((f, t))

class MatReader(object):

    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.file_path = file_path
        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]
        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
        if self.to_float:
            x = x.astype(np.float32)
        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.cuda()
        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

class HybridPhysicsLoss(object):

    def __init__(self, d=2, p=2, size_average=True, reduction=True, cfg=None):
        super(HybridPhysicsLoss, self).__init__()
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        if cfg:
            self.use_constraint = cfg.use_constraint
            self.alpha = cfg.alpha
            self.beta = cfg.beta
            self.dx = cfg.dx
            self.dy = cfg.dy
        else:
            self.use_constraint = True
            self.alpha = 0.1
            self.beta = 0.1
            self.dx = 0.9
            self.dy = 0.9

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        rel_loss = diff_norms / (y_norms + 1e-10)
        if self.alpha > 0 or self.beta > 0:
            dv = self.dx * self.dy
            pred_p = x[..., 0].sum(dim=(1, 2)) * dv
            pred_n = x[..., 1].sum(dim=(1, 2)) * dv
            target_p = y[..., 0].sum(dim=(1, 2)) * dv
            target_n = y[..., 1].sum(dim=(1, 2)) * dv
            p_loss = torch.abs(pred_p - target_p) / (target_p + 1e-10)
            n_loss = torch.abs(pred_n - target_n) / (target_n + 1e-10)
            physics_loss = self.alpha * (p_loss + n_loss)
            total_loss = rel_loss + physics_loss
        else:
            total_loss = rel_loss
        if self.reduction:
            return torch.mean(total_loss) if self.size_average else torch.sum(total_loss)
        return total_loss

    def __call__(self, x, y):
        return self.rel(x, y)

class LpLoss(object):

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class CustomLoss(object):

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(CustomLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y, numconstr):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        lambda0 = 1.0
        lambda1 = 1.0 / 100
        if self.reduction:
            if self.size_average:
                return lambda0 * torch.mean(diff_norms / y_norms) + lambda1 * numconstr
            else:
                return lambda0 * torch.sum(diff_norms / y_norms) + lambda1 * numconstr
        return lambda0 * diff_norms / y_norms + lambda1 * numconstr

    def __call__(self, x, y, numconstr):
        return self.rel(x, y, numconstr)

def iterative_process(model, xx, cfg):
    current_state = xx
    for _ in range(cfg.iteration):
        pred = model(current_state)
        if cfg.predict_residual == 1:
            current_state = current_state + pred
        else:
            current_state = pred
    return current_state

def data_prepare(cfg: TrainConfig, inputpath):
    start_time = time.time()
    Train_dim = int(cfg.TrajNum * (1 - cfg.ratio))
    Val_dim = cfg.TrajNum - Train_dim
    random.seed(cfg.seed_number)
    rand_list = list(range(1, cfg.TrajNum + 1))
    random.shuffle(rand_list)
    Train_list = rand_list[:Train_dim]
    Val_list = rand_list[Train_dim:]
    a_train, u_train, a_val, u_val = ([], [], [], [])
    index = cfg.predstep * cfg.iteration
    print(f'→ Preparing data: {Train_dim} training trajectories, {Val_dim} validation trajectories')
    t0 = time.time()
    for ii in Train_list:
        phis = torch.load(f'{inputpath}/data_2d_{ii:0d}.pt', weights_only=False)
        a_train.append(phis[0:-index])
        u_train.append(phis[index:])
    print(f' Training data loaded in {time.time() - t0:.2f} s')
    t1 = time.time()
    for ii in Val_list:
        phis = torch.load(f'{inputpath}/data_2d_{ii:05d}.pt', weights_only=False)
        a_val.append(phis[0:-index])
        u_val.append(phis[index:])
    print(f' Validation data loaded in {time.time() - t1:.2f} s')
    t2 = time.time()
    a_train = np.concatenate(a_train, axis=0)
    u_train = np.concatenate(u_train, axis=0)
    a_val = np.concatenate(a_val, axis=0)
    u_val = np.concatenate(u_val, axis=0)
    print(f' Data concatenated in {time.time() - t2:.2f} s')
    t3 = time.time()
    indices = np.arange(len(a_train))
    np.random.seed(cfg.seed)
    np.random.shuffle(indices)
    a_train = a_train[indices]
    u_train = u_train[indices]
    indices_val = np.arange(len(a_val))
    np.random.seed(cfg.seed_number + 1)
    np.random.shuffle(indices_val)
    a_val = a_val[indices_val]
    u_val = u_val[indices_val]
    print(f' Data shuffled in {time.time() - t3:.2f} s')
    total_time = time.time() - start_time
    print(f' Total data preparation time: {total_time:.2f} seconds')
    return (a_train, u_train, a_val, u_val)

class SlurmTimeout(Exception):
    pass

def handle_slurm_timeout(signum, frame):
    print('\n[WARNING] SLURM timeout signal received!', flush=True)
    raise SlurmTimeout('Time to save and exit!')

def full_loop(rank, model, optimizer, scheduler, outputpath, split_info, cfg, X_train, Y_train, DN_train, X_val, Y_val, DN_val):
    signal.signal(signal.SIGUSR1, handle_slurm_timeout)
    train_losses = []
    val_losses = []
    train_particle_errors = []
    val_particle_errors = []
    LR = []
    best_val = np.inf
    myloss_hybrid = HybridPhysicsLoss(cfg=cfg)
    base_dir = outputpath
    device = torch.device(f'cuda:{rank}')
    try:
        for ep in range(cfg.num_epochs):
            if dist.is_initialized():
                dist.barrier()
            model.train()
            train_l2_sum = 0.0
            train_dn_sum = 0.0
            ntrain = 0
            local_total = torch.tensor([X_train.size(0)], dtype=torch.long, device=device)
            if dist.is_initialized():
                dist.all_reduce(local_total, op=dist.ReduceOp.MIN)
            safe_total_train = local_total.item()
            indices = torch.randperm(safe_total_train, device=device)
            for start_idx in range(0, safe_total_train, cfg.batch_size):
                batch_indices = indices[start_idx:start_idx + cfg.batch_size]
                xx = X_train[batch_indices]
                yy = Y_train[batch_indices]
                delta_n = DN_train[batch_indices]
                optimizer.zero_grad()
                batch_size = yy.size(0)
                ntrain += batch_size
                train_dn_sum += torch.abs(delta_n).sum().item()
                pred = iterative_process(model, xx, cfg)
                loss = myloss_hybrid(pred, yy)
                train_l2_sum += loss.item()
                loss.backward()
                optimizer.step()
                del xx, yy, pred, loss, delta_n
            if dist.is_initialized():
                train_metrics = torch.tensor([train_l2_sum, train_dn_sum, ntrain], device=device)
                dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
                train_l2_sum, train_dn_sum, ntrain = train_metrics.tolist()
            train_loss_epoch = train_l2_sum / ntrain
            train_dn_epoch = train_dn_sum / ntrain
            train_losses.append(train_loss_epoch)
            train_particle_errors.append(train_dn_epoch)
            if rank == 0:
                print(f'Epoch {ep} | Train Loss: {train_loss_epoch:.6f} | Particle Error: {train_dn_epoch:.6f}', flush=True)
            model.eval()
            val_l2_sum = 0.0
            val_dn_sum = 0.0
            nval = 0
            local_val_total = torch.tensor([X_val.size(0)], dtype=torch.long, device=device)
            if dist.is_initialized():
                dist.all_reduce(local_val_total, op=dist.ReduceOp.MIN)
            safe_total_val = local_val_total.item()
            with torch.no_grad():
                for start_idx in range(0, safe_total_val, cfg.batch_size):
                    end_idx = min(start_idx + cfg.batch_size, safe_total_val)
                    xx = X_val[start_idx:end_idx]
                    yy = Y_val[start_idx:end_idx]
                    delta_n = DN_val[start_idx:end_idx]
                    batch_size = yy.size(0)
                    if batch_size == 0:
                        continue
                    nval += batch_size
                    val_dn_sum += torch.abs(delta_n).sum().item()
                    pred = iterative_process(model, xx, cfg)
                    loss = myloss_hybrid(pred, yy)
                    val_l2_sum += loss.item()
                    del xx, yy, pred, loss, delta_n
            if dist.is_initialized():
                val_metrics = torch.tensor([val_l2_sum, val_dn_sum, nval], device=device)
                dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
                val_l2_sum, val_dn_sum, nval = val_metrics.tolist()
            val_loss_epoch = val_l2_sum / nval
            val_dn_epoch = val_dn_sum / nval
            val_losses.append(val_loss_epoch)
            val_particle_errors.append(val_dn_epoch)
            current_lr = optimizer.param_groups[0]['lr']
            LR.append(current_lr)
            if rank == 0:
                print(f'Epoch {ep} | Valid Loss: {val_loss_epoch:.6f} | Valid Particle Δ: {val_dn_epoch:.6f} | LR: {current_lr:.2e}', flush=True)
            if val_loss_epoch < best_val and ep != cfg.num_epochs - 1:
                best_val = val_loss_epoch
                if rank == 0 and cfg.save_model:
                    checkpoint = {'epoch': ep + 1, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_loss': train_losses, 'val_loss': val_losses, 'train_particle_errors': train_particle_errors, 'val_particle_errors': val_particle_errors, 'Learning_Rate': LR, 'split_info': split_info}
                    torch.save(checkpoint, os.path.join(base_dir, 'model.checkpoint'))
                    print(f' Best model saved at epoch {ep}', flush=True)
            scheduler.step()
            gc.collect()
        if rank == 0 and cfg.save_model:
            print(f'\n[INFO] Training completed normally without timeout. Saving final model...', flush=True)
            checkpoint = {'epoch': cfg.num_epochs, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_loss': train_losses, 'val_loss': val_losses, 'train_particle_errors': train_particle_errors, 'val_particle_errors': val_particle_errors, 'Learning_Rate': LR, 'split_info': split_info}
            final_path = os.path.join(base_dir, 'model_final.checkpoint')
            torch.save(checkpoint, final_path)
            print(f" Final model saved successfully as '{final_path}'", flush=True)
    except SlurmTimeout:
        if rank == 0:
            print(f'\n[TIMEOUT] Emergency save at epoch {ep}...', flush=True)
            checkpoint = {'epoch': ep + 1, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_loss': train_losses, 'val_loss': val_losses, 'train_particle_errors': train_particle_errors, 'val_particle_errors': val_particle_errors, 'Learning_Rate': LR, 'split_info': split_info}
            torch.save(checkpoint, os.path.join(base_dir, 'model_emergency.checkpoint'))
        sys.exit(0)
    return (train_losses, val_losses, LR, train_particle_errors, val_particle_errors)

def parsefile(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    print('\n' + '#' * 50)
    print('Input Parameters (Loaded from YAML):')
    for category, params in config.items():
        if isinstance(params, dict):
            for key, value in params.items():
                print(f'{key}: {value}')
    print('#' * 50 + '\n')
    cfg = TrainConfig(num_nodes=config['hardware']['num_nodes'], n_gpu_per_node=config['hardware']['n_gpu_per_node'], multi_gpu=config['hardware']['multi_gpu'], use_gpu=config['hardware']['use_gpu'], seed=config['training_logic']['seed'], dx=config['data_geometry']['dx'], dy=config['data_geometry']['dy'], dt=config['data_geometry']['dt'], input_shape=config['data_geometry']['input_shape'], num_epochs=config['training_logic']['num_epochs'], batch_size=config['training_logic']['batch_size'], iteration=config['training_logic']['iteration'], pred_step=config['training_logic']['pred_step'], predict_residual=config['training_logic']['predict_residual'], base_lr=config['optimizer']['base_lr'], min_lr=config['optimizer']['min_lr'], weight_decay=config['optimizer']['weight_decay'], scheduler_step=config['optimizer']['scheduler_step'], scheduler_gamma=config['optimizer']['scheduler_gamma'], width=config['model_architecture']['width'], uno_layers=config['model_architecture']['uno_layers'], use_constraint=config['loss_physics']['use_constraint'], alpha=config['loss_physics']['alpha'], beta=config['loss_physics']['beta'], lp_p=config['loss_physics']['lp_p'], val_ratio=config['data_management']['val_ratio'], test_ratio=config['data_management']['test_ratio'], n_select=config['data_management']['n_select'], test_num=config['data_management']['test_num'], project_tag=config['logging_checkpoints']['project_tag'], save_model=config['logging_checkpoints']['save_model'], save_frequency=config['logging_checkpoints']['save_frequency'], checkpoint_dir=config['logging_checkpoints']['checkpoint_dir'], load_model=config['logging_checkpoints']['load_model'], resume_path=config['logging_checkpoints']['resume_path'])
    return cfg

def format_output_log(cfg, output_path):
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
    tag_dir = os.path.join(output_path, f'tag_{cfg.project_tag}')
    os.makedirs(tag_dir, exist_ok=True)
    log_file = os.path.join(tag_dir, f'out_{timestamp}.txt')
    with open(log_file, 'w') as f:
        print('###################################################', file=f)
        print(f'Project Tag: {cfg.project_tag}', file=f)
        print(f'Iteration/Run: {cfg.iteration}', file=f)
        print(f'Timestamp: {timestamp}', file=f)
        print('###################################################\n', file=f)
        print(f'Number of UNO layers: {cfg.uno_layers}', file=f)
        print(f'Model width: {cfg.width}', file=f)
        print(f'Prediction step (indices): {cfg.pred_step}', file=f)
        print(f'Prediction step (fm/c): {cfg.dt * cfg.pred_step:.2f}', file=f)
        print('###################################################\n', file=f)
        print(f'Validation ratio: {cfg.val_ratio}', file=f)
        print(f'Batch size (per GPU): {cfg.batch_size}', file=f)
        print(f'Number of trajectories (n_select): {cfg.n_select}', file=f)
        print('###################################################\n', file=f)
        print(f'Number of epochs: {cfg.num_epochs}', file=f)
        print(f'Scheduler Gamma: {cfg.scheduler_gamma}', file=f)
        print(f'Scheduler Step Size: {cfg.scheduler_step}', file=f)
        print(f'Weight decay: {cfg.weight_decay}', file=f)
        print(f'Base learning rate: {cfg.base_lr}', file=f)
        print('###################################################\n', file=f)
        print(f'Use Physics Constraint: {cfg.use_constraint}', file=f)
        print(f'Alpha (Particle weight): {cfg.alpha}', file=f)
        print(f'Beta (Physics weight): {cfg.beta}', file=f)
        print('###################################################', file=f)
    print(f'Log saved to: {log_file}')