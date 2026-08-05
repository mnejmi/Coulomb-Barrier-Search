import torch
import numpy as np
import os
from dataclasses import dataclass
from torch.utils.data import IterableDataset, get_worker_info
from pathlib import Path
import yaml
import signal
import sys
import inspect
from typing import List
import torch.distributed as dist

class SlurmTimeout(Exception):
    pass

def handle_slurm_timeout(signum, frame):
    print('\n[WARNING] SLURM timeout signal received!', flush=True)
    raise SlurmTimeout('Time to save and exit!')

@dataclass
class TrainConfig:
    num_nodes: int
    n_gpu_per_node: int
    multi_gpu: bool
    use_gpu: bool
    seed: int
    trajectories_seed: int
    cpu_per_task: int
    dx: float
    dy: float
    dt: float
    input_shape: List[int]
    num_epochs: int
    batch_size: int
    iteration: int
    pred_step: int
    predict_residual: bool
    curriculum_epochs: int
    start_iteration: int
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

class TrajectoryIterableDataset(IterableDataset):

    def __init__(self, data_dir, cfg, split='train', rank=0, world_size=1):
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.predict_residual = cfg.predict_residual
        all_files = sorted(list(self.data_dir.glob('*.pt')))
        if len(all_files) == 0:
            raise RuntimeError(f' No .pt files found in {self.data_dir}')
        rng = np.random.default_rng(cfg.seed)
        rng.shuffle(all_files)
        if cfg.n_select is not None and cfg.n_select < len(all_files):
            all_files = all_files[:cfg.n_select]
        total = len(all_files)
        n_val = int(total * cfg.val_ratio)
        n_test = int(total * cfg.test_ratio)
        n_train = total - n_val - n_test
        if split == 'train':
            split_files = all_files[:n_train]
        elif split == 'val':
            split_files = all_files[n_train:n_train + n_val]
        else:
            split_files = all_files[n_train + n_val:]
        chunk_size = len(split_files) // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank != world_size - 1 else len(split_files)
        self.files = split_files[start_idx:end_idx]

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_files = self.files
        else:
            per_worker = int(np.ceil(len(self.files) / worker_info.num_workers))
            worker_id = worker_info.id
            worker_files = self.files[worker_id * per_worker:(worker_id + 1) * per_worker]
        rng = np.random.default_rng()
        rng.shuffle(worker_files)
        for file_path in worker_files:
            try:
                data = torch.load(file_path, map_location='cpu', weights_only=True)
                T = data.shape[0]
                current_iter = getattr(self.cfg, 'current_iteration', self.cfg.iteration)
                dynamic_max_index = self.cfg.pred_step * current_iter
                if T > dynamic_max_index:
                    indices = np.arange(T - dynamic_max_index)
                    rng.shuffle(indices)
                    for t_idx in indices:
                        frame_x = data[t_idx].float()
                        y_list = []
                        for i in range(1, current_iter + 1):
                            offset = i * self.cfg.pred_step
                            frame_future = data[t_idx + offset].float()
                            if self.predict_residual:
                                frame_y = frame_future - frame_x
                                frame_x = frame_future.clone()
                            else:
                                frame_y = frame_future
                            y_list.append(frame_y)
                        frame_y_seq = torch.stack(y_list, dim=0)
                        yield (data[t_idx].float(), frame_y_seq)
                del data
            except Exception as e:
                print(f'️ Skipping corrupted file {file_path.name}: {e}')

class RelativeLoss(object):

    def __init__(self, p=2, size_average=True, reduction=True, cfg=None):
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.predict_residual = getattr(cfg, 'predict_residual', False) if cfg else False

    def __call__(self, x, y):
        if x.dim() == 5:
            B, T, H, W, C = x.shape
            x = x.reshape(B * T, H, W, C)
            y = y.reshape(B * T, H, W, C)
        num_examples = x.size(0)
        x_flat = x.reshape(num_examples, -1)
        y_flat = y.reshape(num_examples, -1)
        diff_norms = torch.norm(x_flat - y_flat, self.p, 1)
        if self.predict_residual:
            rel_loss = diff_norms
        else:
            y_norms = torch.norm(y_flat, self.p, 1)
            rel_loss = diff_norms / (y_norms + 1e-10)
        if self.reduction:
            if self.size_average:
                return torch.mean(rel_loss)
            return torch.sum(rel_loss)
        return rel_loss

def full_loop(rank, device, model, optimizer, scheduler, outputpath, split_info, cfg, train_loader, val_loader, start_epoch=0, history=None):
    signal.signal(signal.SIGUSR1, handle_slurm_timeout)
    is_main = rank == 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if history is not None:
        train_losses = history.get('train_loss', [])
        val_losses = history.get('val_loss', [])
        LR = history.get('Learning_Rate', [])
        best_val = min(val_losses) if len(val_losses) > 0 else np.inf
    else:
        train_losses, val_losses, LR = ([], [], [])
        best_val = np.inf
    myloss = RelativeLoss(cfg=cfg)
    scaler = torch.amp.GradScaler('cuda')
    try:
        cfg.current_iteration = getattr(cfg, 'start_iteration', 1)
        for ep in range(start_epoch, cfg.num_epochs):
            if dist.is_initialized():
                dist.barrier()
            if getattr(cfg, 'curriculum_epochs', 0) > 0:
                steps_increased = ep // cfg.curriculum_epochs
                new_iteration = min(cfg.start_iteration + steps_increased, cfg.iteration)
                if new_iteration != cfg.current_iteration:
                    cfg.current_iteration = new_iteration
                    if is_main:
                        print(f'\n [CURRICULUM UPGRADE] Increasing rollout to {cfg.current_iteration} steps!\n')
            model.train()
            train_l2_sum, ntrain = (0.0, 0)
            train_iter = iter(train_loader)
            while True:
                try:
                    xx, yy = next(train_iter)
                    has_data = torch.tensor([1], device=device)
                except StopIteration:
                    has_data = torch.tensor([0], device=device)
                    xx = torch.zeros(1, device=device)
                    yy = torch.zeros(1, device=device)
                if dist.is_initialized():
                    dist.all_reduce(has_data)
                    if has_data.item() < world_size:
                        break
                xx = xx.to(device, non_blocking=True)
                yy = yy.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                batch_size = yy.size(0)
                ntrain += batch_size
                with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                    pred = model(xx)
                    loss = myloss(pred, yy)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_l2_sum += loss.item() * batch_size
            if dist.is_initialized():
                train_metrics = torch.tensor([train_l2_sum, ntrain], device=device)
                dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
                train_l2_sum, ntrain = train_metrics.tolist()
            train_loss_epoch = train_l2_sum / ntrain
            train_losses.append(train_loss_epoch)
            if is_main:
                print(f'Epoch {ep} | Train Loss: {train_loss_epoch:.6f} |', end=' ')
            model.eval()
            val_l2_sum, nval = (0.0, 0)
            val_iter = iter(val_loader)
            with torch.no_grad():
                while True:
                    try:
                        xx, yy = next(val_iter)
                        has_data = torch.tensor([1], device=device)
                    except StopIteration:
                        has_data = torch.tensor([0], device=device)
                        xx = torch.zeros(1, device=device)
                        yy = torch.zeros(1, device=device)
                    if dist.is_initialized():
                        dist.all_reduce(has_data)
                        if has_data.item() < world_size:
                            break
                    xx = xx.to(device, non_blocking=True)
                    yy = yy.to(device, non_blocking=True)
                    batch_size = yy.size(0)
                    nval += batch_size
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                        pred = model.module(xx)
                        loss = myloss(pred, yy)
                    val_l2_sum += loss.item() * batch_size
            if dist.is_initialized():
                val_metrics = torch.tensor([val_l2_sum, nval], device=device)
                dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
                val_l2_sum, nval = val_metrics.tolist()
            val_loss_epoch = val_l2_sum / nval
            val_losses.append(val_loss_epoch)
            current_lr = optimizer.param_groups[0]['lr']
            LR.append(current_lr)
            if is_main:
                print(f'Valid Loss: {val_loss_epoch:.6f} | LR: {current_lr:.2e}', flush=True)
            if is_main and val_loss_epoch < best_val and (ep != cfg.num_epochs - 1):
                best_val = val_loss_epoch
                if cfg.save_model:
                    checkpoint = {'epoch': ep + 1, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_loss': train_losses, 'val_loss': val_losses, 'Learning_Rate': LR, 'split_info': split_info}
                    torch.save(checkpoint, os.path.join(outputpath, 'model.checkpoint'))
            scheduler.step()
        if is_main and cfg.save_model:
            checkpoint = {'epoch': cfg.num_epochs, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_loss': train_losses, 'val_loss': val_losses, 'Learning_Rate': LR, 'split_info': split_info}
            torch.save(checkpoint, os.path.join(outputpath, 'model_final.checkpoint'))
    except SlurmTimeout:
        if is_main:
            checkpoint = {'epoch': ep + 1, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_loss': train_losses, 'val_loss': val_losses, 'Learning_Rate': LR, 'split_info': split_info}
            torch.save(checkpoint, os.path.join(outputpath, 'model_emergency.checkpoint'))
        sys.exit(0)
    return (train_losses, val_losses, LR)