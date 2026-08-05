import numpy as np
import torch
import torch.nn as nn
import math
import os
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from Adam import Adam
from neural_operator import UNetNeuralOperator
from src.utilities3 import TrainConfig, TrajectoryIterableDataset, full_loop, load_config

def setup_ddp():
    rank = int(os.environ.get('SLURM_PROCID', 0))
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    return (rank, local_rank, world_size)

class IterativeModelWrapper(nn.Module):

    def __init__(self, core_model, cfg):
        super().__init__()
        self.core_model = core_model
        self.cfg = cfg

    def forward(self, xx):
        current_state = xx
        predictions = []
        current_iter = getattr(self.cfg, 'current_iteration', self.cfg.iteration)
        for _ in range(current_iter):
            step_pred = self.core_model(current_state)
            if self.cfg.predict_residual:
                current_state = current_state + step_pred
                predictions.append(step_pred)
            else:
                current_state = step_pred
                predictions.append(current_state)
        return torch.stack(predictions, dim=1)

class LogisticLRScheduler(_LRScheduler):

    def __init__(self, optimizer, num_epochs, initial_lr, final_lr, last_epoch=-1):
        self.num_epochs = num_epochs
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        mid = self.num_epochs / 2
        k = self.num_epochs / 10
        lr = self.final_lr + (self.initial_lr - self.final_lr) / (1 + math.exp((step - mid) / k))
        return [lr for _ in self.optimizer.param_groups]

def main(cfg, checkpoint_dir, data_path):
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    is_main = rank == 0
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    if is_main:
        print('\n' + '=' * 60)
        print(f' DDP TRAINING START | PURE ML STEP-BY-STEP')
        print(f'World size: {world_size}')
        print('=' * 60 + '\n')
    NCPU = cfg.cpu_per_task
    train_dataset = TrajectoryIterableDataset(data_dir=data_path, cfg=cfg, split='train', rank=rank, world_size=world_size)
    val_dataset = TrajectoryIterableDataset(data_dir=data_path, cfg=cfg, split='val', rank=rank, world_size=world_size)
    test_dataset = TrajectoryIterableDataset(data_dir=data_path, cfg=cfg, split='test', rank=rank, world_size=world_size)
    split_info = [str(f.name) for f in test_dataset.files]
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=NCPU, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True, num_workers=max(1, NCPU // 2), pin_memory=True, persistent_workers=True, prefetch_factor=4)
    base_model = UNetNeuralOperator(in_channels=8, width=cfg.width).to(device)
    model = IterativeModelWrapper(base_model, cfg)
    optimizer = Adam(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay, amsgrad=False)
    start_epoch = 0
    history = None
    if cfg.load_model and os.path.exists(cfg.resume_path):
        if is_main:
            print(f' Loading checkpoint {cfg.resume_path}')
        checkpoint = torch.load(cfg.resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        history = checkpoint
        if is_main:
            print(f' Resumed from epoch {start_epoch}')
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    scheduler = LogisticLRScheduler(optimizer, num_epochs=cfg.num_epochs, initial_lr=cfg.base_lr, final_lr=cfg.min_lr, last_epoch=start_epoch - 1)
    train_losses, val_losses, LR = full_loop(rank=rank, device=device, model=model, optimizer=optimizer, scheduler=scheduler, outputpath=checkpoint_dir, split_info=split_info, cfg=cfg, train_loader=train_loader, val_loader=val_loader, start_epoch=start_epoch, history=history)
    dist.destroy_process_group()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    main(cfg, args.checkpoint_dir, args.data_path)