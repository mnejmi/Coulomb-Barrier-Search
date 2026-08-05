import argparse
import os
import time
import math
import torch
import torch.nn as nn
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning.loggers import CSVLogger
import yaml
import torch.nn.functional as F
import warnings
from utilities3 import LpLoss, CustomLoss
from physics_dataset import data_prepare
from torch.utils.data import TensorDataset, DataLoader
from Transformer_UNO_v5 import Transformer_UNO_v5

class PhaseAwareCheckpoint(Callback):
    PHASES = [(0, 199, 'best_phase1'), (200, 399, 'best_phase2'), (400, 999, 'best_phase3')]

    def __init__(self, dirpath: str):
        super().__init__()
        self.dirpath = dirpath
        self._phase_best: dict = {}
        self._abs_best: float = float('inf')

    def _phase_key(self, epoch: int):
        for start, end, key in self.PHASES:
            if start <= epoch <= end:
                return key
        return None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is None:
            return
        val_loss = float(val_loss)
        epoch = trainer.current_epoch
        key = self._phase_key(epoch)
        if key is not None:
            if val_loss < self._phase_best.get(key, float('inf')):
                self._phase_best[key] = val_loss
                ckpt_path = os.path.join(self.dirpath, f'{key}.ckpt')
                trainer.save_checkpoint(ckpt_path)
                print(f' [{key}] New best val_loss={val_loss:.6f} at epoch {epoch} → {ckpt_path}')
        if val_loss < self._abs_best:
            self._abs_best = val_loss
            ckpt_path = os.path.join(self.dirpath, 'best_absolute.ckpt')
            trainer.save_checkpoint(ckpt_path)
            print(f' [best_absolute] New best val_loss={val_loss:.6f} at epoch {epoch} → {ckpt_path}')
warnings.filterwarnings('ignore', category=UserWarning)

class TrainConfig:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class PhysicsInformedLoss(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lploss = LpLoss(size_average=False)
        self.alpha = nn.Parameter(torch.tensor(getattr(cfg, 'alpha', 0.0), dtype=torch.float32), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(getattr(cfg, 'beta', 1.0), dtype=torch.float32), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(getattr(cfg, 'gamma', 0.0), dtype=torch.float32), requires_grad=False)
        self.delta = nn.Parameter(torch.tensor(getattr(cfg, 'delta', 0.0), dtype=torch.float32), requires_grad=False)
        self.omega = nn.Parameter(torch.tensor(getattr(cfg, 'omega', 0.0), dtype=torch.float32), requires_grad=False)
        stats_path = getattr(cfg, 'stats_path', '/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/TDHF90_NORMALIZED/global_normalization_stats.pt')
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location='cpu', weights_only=True)
            self.register_buffer('m_a', stats['m'].float())
            self.register_buffer('std_a', stats['std'].float())
        else:
            self.register_buffer('m_a', torch.tensor(0.0))
            self.register_buffer('std_a', torch.tensor(1.0))

    def forward(self, x_in, pred, y):
        batch_size = pred.shape[0]
        iteration = pred.shape[1]
        loss = 0.0
        acc_l2 = 0.0
        acc_continuity = 0.0
        acc_mass = 0.0
        acc_momentum = 0.0
        acc_wass = 0.0
        for step in range(iteration):
            pred_step = pred[:, step]
            y_step = y[:, step]
            if step == 0:
                prev_state = x_in[..., :6]
            else:
                prev_state = pred[:, step - 1][..., :6]
            pred_flat = pred_step.reshape(batch_size, -1)
            y_flat = y_step.reshape(batch_size, -1)
            step_l2 = self.beta * self.lploss(pred_flat, y_flat)
            step_loss = step_l2
            acc_l2 = acc_l2 + step_l2
            if self.alpha > 0.0 or self.gamma > 0.0 or self.delta > 0.0 or (self.omega > 0.0):
                rho_p_t = prev_state[:, :, :, 0] * self.std_a[0] + self.m_a[0]
                rho_p_t1 = pred_step[:, :, :, 0] * self.std_a[0] + self.m_a[0]
                jx_p_t = prev_state[:, :, :, 2] * self.std_a[2] + self.m_a[2]
                jx_p_t1 = pred_step[:, :, :, 2] * self.std_a[2] + self.m_a[2]
                jy_p_t = prev_state[:, :, :, 4] * self.std_a[4] + self.m_a[4]
                jy_p_t1 = pred_step[:, :, :, 4] * self.std_a[4] + self.m_a[4]
                rho_n_t = prev_state[:, :, :, 1] * self.std_a[1] + self.m_a[1]
                rho_n_t1 = pred_step[:, :, :, 1] * self.std_a[1] + self.m_a[1]
                jx_n_t = prev_state[:, :, :, 3] * self.std_a[3] + self.m_a[3]
                jx_n_t1 = pred_step[:, :, :, 3] * self.std_a[3] + self.m_a[3]
                jy_n_t = prev_state[:, :, :, 5] * self.std_a[5] + self.m_a[5]
                jy_n_t1 = pred_step[:, :, :, 5] * self.std_a[5] + self.m_a[5]
                loss_physics = 0.0
                if self.alpha > 0.0:
                    pred_step_cfg = getattr(self.cfg, 'pred_step', 1)
                    dt_cfg = getattr(self.cfg, 'dt', 9.0)
                    dx_cfg = getattr(self.cfg, 'dx', 0.9)
                    c_grid = dt_cfg * pred_step_cfg / dx_cfg
                    djx_n_t_dx = (-jx_n_t[:, 2:-2, 4:] + 8 * jx_n_t[:, 2:-2, 3:-1] - 8 * jx_n_t[:, 2:-2, 1:-3] + jx_n_t[:, 2:-2, :-4]) / 12.0
                    djx_n_t1_dx = (-jx_n_t1[:, 2:-2, 4:] + 8 * jx_n_t1[:, 2:-2, 3:-1] - 8 * jx_n_t1[:, 2:-2, 1:-3] + jx_n_t1[:, 2:-2, :-4]) / 12.0
                    djy_n_t_dy = (-jy_n_t[:, 4:, 2:-2] + 8 * jy_n_t[:, 3:-1, 2:-2] - 8 * jy_n_t[:, 1:-3, 2:-2] + jy_n_t[:, :-4, 2:-2]) / 12.0
                    djy_n_t1_dy = (-jy_n_t1[:, 4:, 2:-2] + 8 * jy_n_t1[:, 3:-1, 2:-2] - 8 * jy_n_t1[:, 1:-3, 2:-2] + jy_n_t1[:, :-4, 2:-2]) / 12.0
                    div_J_n_t = djx_n_t_dx + djy_n_t_dy
                    div_J_n_t1 = djx_n_t1_dx + djy_n_t1_dy
                    div_J_n_mid = (div_J_n_t + div_J_n_t1) / 2.0
                    d_rho_n = rho_n_t1 - rho_n_t
                    error_n_phys = d_rho_n[:, 2:-2, 2:-2] - -c_grid * div_J_n_mid
                    error_n_norm = error_n_phys / self.std_a[1]
                    loss_n = F.mse_loss(error_n_norm, torch.zeros_like(error_n_norm))
                    djx_p_t_dx = (-jx_p_t[:, 2:-2, 4:] + 8 * jx_p_t[:, 2:-2, 3:-1] - 8 * jx_p_t[:, 2:-2, 1:-3] + jx_p_t[:, 2:-2, :-4]) / 12.0
                    djx_p_t1_dx = (-jx_p_t1[:, 2:-2, 4:] + 8 * jx_p_t1[:, 2:-2, 3:-1] - 8 * jx_p_t1[:, 2:-2, 1:-3] + jx_p_t1[:, 2:-2, :-4]) / 12.0
                    djy_p_t_dy = (-jy_p_t[:, 4:, 2:-2] + 8 * jy_p_t[:, 3:-1, 2:-2] - 8 * jy_p_t[:, 1:-3, 2:-2] + jy_p_t[:, :-4, 2:-2]) / 12.0
                    djy_p_t1_dy = (-jy_p_t1[:, 4:, 2:-2] + 8 * jy_p_t1[:, 3:-1, 2:-2] - 8 * jy_p_t1[:, 1:-3, 2:-2] + jy_p_t1[:, :-4, 2:-2]) / 12.0
                    div_J_p_t = djx_p_t_dx + djy_p_t_dy
                    div_J_p_t1 = djx_p_t1_dx + djy_p_t1_dy
                    div_J_p_mid = (div_J_p_t + div_J_p_t1) / 2.0
                    d_rho_p = rho_p_t1 - rho_p_t
                    error_p_phys = d_rho_p[:, 2:-2, 2:-2] - -c_grid * div_J_p_mid
                    error_p_norm = error_p_phys / self.std_a[0]
                    loss_p = F.mse_loss(error_p_norm, torch.zeros_like(error_p_norm))
                    step_continuity = self.alpha * (loss_n + loss_p)
                    loss_physics = loss_physics + step_continuity
                    acc_continuity = acc_continuity + step_continuity
                if self.gamma > 0.0:
                    dx_cfg = getattr(self.cfg, 'dx', 0.9)
                    dy_cfg = getattr(self.cfg, 'dy', 0.9)
                    dV = dx_cfg * dy_cfg
                    mass_n_t = torch.sum(rho_n_t, dim=(1, 2)) * dV
                    mass_n_t1 = torch.sum(rho_n_t1, dim=(1, 2)) * dV
                    mass_p_t = torch.sum(rho_p_t, dim=(1, 2)) * dV
                    mass_p_t1 = torch.sum(rho_p_t1, dim=(1, 2)) * dV
                    error_mass_n_norm = (mass_n_t1 - mass_n_t) / (self.std_a[1] * dV * (56 * 56))
                    error_mass_p_norm = (mass_p_t1 - mass_p_t) / (self.std_a[0] * dV * (56 * 56))
                    loss_mass = F.mse_loss(error_mass_n_norm, torch.zeros_like(error_mass_n_norm)) + F.mse_loss(error_mass_p_norm, torch.zeros_like(error_mass_p_norm))
                    step_mass = self.gamma * loss_mass
                    loss_physics = loss_physics + step_mass
                    acc_mass = acc_mass + step_mass
                if self.delta > 0.0:
                    px_n_t = torch.sum(jx_n_t, dim=(1, 2))
                    px_n_t1 = torch.sum(jx_n_t1, dim=(1, 2))
                    py_n_t = torch.sum(jy_n_t, dim=(1, 2))
                    py_n_t1 = torch.sum(jy_n_t1, dim=(1, 2))
                    px_p_t = torch.sum(jx_p_t, dim=(1, 2))
                    px_p_t1 = torch.sum(jx_p_t1, dim=(1, 2))
                    py_p_t = torch.sum(jy_p_t, dim=(1, 2))
                    py_p_t1 = torch.sum(jy_p_t1, dim=(1, 2))
                    error_px_n_norm = (px_n_t1 - px_n_t) / (self.std_a[3] * (56 * 56))
                    error_py_n_norm = (py_n_t1 - py_n_t) / (self.std_a[5] * (56 * 56))
                    error_px_p_norm = (px_p_t1 - px_p_t) / (self.std_a[2] * (56 * 56))
                    error_py_p_norm = (py_p_t1 - py_p_t) / (self.std_a[4] * (56 * 56))
                    loss_mom = F.mse_loss(error_px_n_norm, torch.zeros_like(error_px_n_norm)) + F.mse_loss(error_py_n_norm, torch.zeros_like(error_py_n_norm)) + F.mse_loss(error_px_p_norm, torch.zeros_like(error_px_p_norm)) + F.mse_loss(error_py_p_norm, torch.zeros_like(error_py_p_norm))
                    step_momentum = self.delta * loss_mom
                    loss_physics = loss_physics + step_momentum
                    acc_momentum = acc_momentum + step_momentum
                if self.omega > 0.0:
                    rho_n_true = y_step[:, :, :, 1] * self.std_a[1] + self.m_a[1]
                    rho_p_true = y_step[:, :, :, 0] * self.std_a[0] + self.m_a[0]
                    rho_n_t1_pos = F.relu(rho_n_t1) + 1e-08
                    rho_p_t1_pos = F.relu(rho_p_t1) + 1e-08
                    rho_n_true_pos = F.relu(rho_n_true) + 1e-08
                    rho_p_true_pos = F.relu(rho_p_true) + 1e-08
                    marg_x_n_pred = torch.sum(rho_n_t1_pos, dim=1)
                    marg_x_n_true = torch.sum(rho_n_true_pos, dim=1)
                    marg_x_p_pred = torch.sum(rho_p_t1_pos, dim=1)
                    marg_x_p_true = torch.sum(rho_p_true_pos, dim=1)
                    marg_y_n_pred = torch.sum(rho_n_t1_pos, dim=2)
                    marg_y_n_true = torch.sum(rho_n_true_pos, dim=2)
                    marg_y_p_pred = torch.sum(rho_p_t1_pos, dim=2)
                    marg_y_p_true = torch.sum(rho_p_true_pos, dim=2)
                    cdf_x_n_pred = torch.cumsum(marg_x_n_pred, dim=1)
                    cdf_x_n_true = torch.cumsum(marg_x_n_true, dim=1)
                    cdf_x_p_pred = torch.cumsum(marg_x_p_pred, dim=1)
                    cdf_x_p_true = torch.cumsum(marg_x_p_true, dim=1)
                    cdf_y_n_pred = torch.cumsum(marg_y_n_pred, dim=1)
                    cdf_y_n_true = torch.cumsum(marg_y_n_true, dim=1)
                    cdf_y_p_pred = torch.cumsum(marg_y_p_pred, dim=1)
                    cdf_y_p_true = torch.cumsum(marg_y_p_true, dim=1)
                    cdf_x_n_pred = cdf_x_n_pred / cdf_x_n_pred[:, -1:]
                    cdf_x_n_true = cdf_x_n_true / cdf_x_n_true[:, -1:]
                    cdf_x_p_pred = cdf_x_p_pred / cdf_x_p_pred[:, -1:]
                    cdf_x_p_true = cdf_x_p_true / cdf_x_p_true[:, -1:]
                    cdf_y_n_pred = cdf_y_n_pred / cdf_y_n_pred[:, -1:]
                    cdf_y_n_true = cdf_y_n_true / cdf_y_n_true[:, -1:]
                    cdf_y_p_pred = cdf_y_p_pred / cdf_y_p_pred[:, -1:]
                    cdf_y_p_true = cdf_y_p_true / cdf_y_p_true[:, -1:]
                    loss_wass = torch.mean(torch.abs(cdf_x_n_pred - cdf_x_n_true)) + torch.mean(torch.abs(cdf_x_p_pred - cdf_x_p_true)) + torch.mean(torch.abs(cdf_y_n_pred - cdf_y_n_true)) + torch.mean(torch.abs(cdf_y_p_pred - cdf_y_p_true))
                    step_wass = self.omega * loss_wass
                    loss_physics = loss_physics + step_wass
                    acc_wass = acc_wass + step_wass
                step_loss = step_loss + loss_physics
            loss += step_loss
        total = loss / iteration
        return {'loss': total, 'loss_l2': acc_l2 / iteration, 'loss_continuity': acc_continuity / iteration, 'loss_mass': acc_mass / iteration, 'loss_momentum': acc_momentum / iteration, 'loss_wass': acc_wass / iteration}

class UNOT_Lightning(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(vars(cfg))
        self.cfg = cfg
        self.model = Transformer_UNO_v5(8, cfg.width, dropout_rate=getattr(cfg, 'dropout_rate', 0.0))
        self.criterion = PhysicsInformedLoss(cfg)

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        if self.current_epoch < 200:
            self.current_iteration = 1
        elif self.current_epoch < 400:
            self.current_iteration = 3
        else:
            self.current_iteration = 6
        if self.current_epoch in [0, 200, 400]:
            print(f' Epoch {self.current_epoch}: SGDR Warm Restart Phase! Training with {self.current_iteration} autoregressive iterations!')

    def _iterative_process(self, x, num_iterations):
        batch_size = x.shape[0]
        preds = []
        curr_x = x
        memory_buffer = []
        for _ in range(num_iterations):
            if getattr(self.cfg, 'predict_residual', False):
                im_delta, memory_buffer = self.model(curr_x, memory_buffer)
                im_r = curr_x + im_delta
            else:
                im_r, memory_buffer = self.model(curr_x, memory_buffer)
            preds.append(im_r)
            curr_x = im_r
        return torch.stack(preds, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]
        num_iters = getattr(self, 'current_iteration', 1)
        pred = self._iterative_process(x, num_iters)
        loss_dict = self.criterion(x, pred, y[:, :num_iters])
        loss = loss_dict['loss']
        log_kwargs = dict(on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log('train_loss', loss / batch_size, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('train_loss_l2', loss_dict['loss_l2'] / batch_size, **log_kwargs)
        self.log('train_loss_continuity', loss_dict['loss_continuity'] / batch_size, **log_kwargs)
        self.log('train_loss_mass', loss_dict['loss_mass'] / batch_size, **log_kwargs)
        self.log('train_loss_momentum', loss_dict['loss_momentum'] / batch_size, **log_kwargs)
        self.log('train_loss_wass', loss_dict['loss_wass'] / batch_size, **log_kwargs)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]
        pred = self._iterative_process(x, self.cfg.iteration)
        loss_dict = self.criterion(x, pred, y)
        val_loss = loss_dict['loss']
        self.log('val_loss', val_loss / batch_size, prog_bar=True, sync_dist=True, batch_size=batch_size)
        log_kwargs = dict(on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        self.log('val_loss_l2', loss_dict['loss_l2'] / batch_size, **log_kwargs)
        self.log('val_loss_continuity', loss_dict['loss_continuity'] / batch_size, **log_kwargs)
        self.log('val_loss_mass', loss_dict['loss_mass'] / batch_size, **log_kwargs)
        self.log('val_loss_momentum', loss_dict['loss_momentum'] / batch_size, **log_kwargs)
        self.log('val_loss_wass', loss_dict['loss_wass'] / batch_size, **log_kwargs)
        return val_loss

    def on_load_checkpoint(self, checkpoint):
        if 'state_dict' in checkpoint:
            keys_to_delete = [k for k in checkpoint['state_dict'].keys() if '_grid_cache' in k]
            for k in keys_to_delete:
                del checkpoint['state_dict'][k]
        if 'callbacks' in checkpoint:
            keys_to_delete = [k for k in checkpoint['callbacks'].keys() if 'EarlyStopping' in k]
            for k in keys_to_delete:
                del checkpoint['callbacks'][k]

    def on_train_start(self):
        schedulers = self.trainer.lr_scheduler_configs
        if schedulers:
            s = schedulers[0].scheduler
            if getattr(s, 'last_epoch', 0) < self.current_epoch:
                s.step(self.current_epoch)
        if schedulers:
            s = schedulers[0].scheduler
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group['lr'] = s.get_last_lr()[0]

    def on_train_epoch_end(self):
        if hasattr(self, 'manual_scheduler'):
            self.manual_scheduler.step()
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.base_lr, weight_decay=self.cfg.weight_decay, amsgrad=False)
        optimizer.add_param_group({'params': self.criterion.parameters(), 'lr': self.cfg.base_lr})
        t0 = getattr(self.cfg, 'lr_T_0', 200)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=1, eta_min=getattr(self.cfg, 'min_lr', 1e-05))
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

def setup_data_loaders(cfg, inputpath):
    train_dataset, valid_dataset = data_prepare(cfg, inputpath)
    if getattr(cfg, 'cpu_per_task', None) is not None:
        optimal_workers = max(1, int(cfg.cpu_per_task) - 1)
    elif 'SLURM_CPUS_PER_TASK' in os.environ:
        optimal_workers = max(1, int(os.environ['SLURM_CPUS_PER_TASK']) - 1)
    else:
        optimal_workers = getattr(cfg, 'num_workers', 1)
    print(f' Dynamically configured DataLoader with num_workers = {optimal_workers}')
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=optimal_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=max(1, optimal_workers // 2))
    return (train_loader, valid_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    cfg = TrainConfig(**config_dict)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_loader, valid_loader = setup_data_loaders(cfg, args.data_path)
    model = UNOT_Lightning(cfg)
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir, filename='model_final', save_top_k=1, monitor='val_loss', mode='min', save_last=True, save_on_train_epoch_end=True, every_n_epochs=1)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    phase_checkpoint = PhaseAwareCheckpoint(dirpath=args.checkpoint_dir)
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=1e-05, patience=500, mode='min')
    csv_logger = CSVLogger(save_dir=args.checkpoint_dir, name='logs')
    if getattr(cfg, 'multi_gpu', False):
        strategy = 'ddp_find_unused_parameters_true'
        devices = cfg.n_gpu_per_node
        num_nodes = cfg.num_nodes
    else:
        strategy = 'auto'
        devices = 1 if not cfg.use_gpu else cfg.n_gpu_per_node
        num_nodes = 1
    avg_frames = 130
    global_batch_size = cfg.batch_size * devices * num_nodes
    limit_train = max(1, int(len(train_loader.dataset.file_list) * avg_frames / global_batch_size))
    limit_val = max(1, int(len(valid_loader.dataset.file_list) * avg_frames / global_batch_size))
    print(f' IterableDataset configured for exactly {limit_train} train steps and {limit_val} val steps per epoch.')
    trainer = Trainer(max_epochs=cfg.num_epochs, max_time='99:00:00:00', accelerator='gpu' if cfg.use_gpu else 'cpu', devices=devices, num_nodes=num_nodes, strategy=strategy, callbacks=[checkpoint_callback, phase_checkpoint, lr_monitor, early_stop_callback], logger=csv_logger, limit_train_batches=limit_train, limit_val_batches=limit_val, use_distributed_sampler=False, fast_dev_run=False)
    last_ckpt_path = os.path.join(args.checkpoint_dir, 'last.ckpt')
    if getattr(cfg, 'load_model', False) and os.path.exists(last_ckpt_path):
        print(f'Resuming training from {last_ckpt_path}')
        trainer.fit(model, train_loader, valid_loader, ckpt_path=last_ckpt_path)
    else:
        trainer.fit(model, train_loader, valid_loader)
if __name__ == '__main__':
    main()