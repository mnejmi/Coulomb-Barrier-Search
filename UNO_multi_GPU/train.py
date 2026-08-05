import argparse
import os
import time
import math
import torch
import torch.nn as nn
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
import yaml
import torch.nn.functional as F
import warnings
try:
    from utilities3 import LpLoss, CustomLoss
except ImportError:
    from .utilities3 import LpLoss, CustomLoss

try:
    from physics_dataset import data_prepare
except ImportError:
    from .physics_dataset import data_prepare

try:
    from UNO_demo_rhocur import UNO_demo
except ImportError:
    from .UNO_demo_rhocur import UNO_demo
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
        stats_path = getattr(cfg, 'stats_path', '')
        if not os.path.exists(stats_path):
            repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            stats_path = os.path.join(repo_dir, 'local_data', 'global_normalization_stats.pt')
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location='cpu')
            self.register_buffer('m_a', stats['m'].float())
            self.register_buffer('std_a', stats['std'].float())
        else:
            self.register_buffer('m_a', torch.zeros(6, dtype=torch.float32))
            self.register_buffer('std_a', torch.ones(6, dtype=torch.float32))

    def forward(self, x_in, pred, y):
        batch_size = pred.shape[0]
        iteration = pred.shape[1]
        loss = 0
        for step in range(iteration):
            pred_step = pred[:, step]
            y_step = y[:, step]
            if step == 0:
                prev_state = x_in[..., :6]
            else:
                prev_state = pred[:, step - 1][..., :6]
            pred_flat = pred_step.reshape(batch_size, -1)
            y_flat = y_step.reshape(batch_size, -1)
            step_loss = self.beta * self.lploss(pred_flat, y_flat)
            if self.alpha > 0.0 or self.gamma > 0.0 or self.delta > 0.0 or (self.omega > 0.0):
                rho_n_t = prev_state[:, :, :, 1] * self.std_a[1] + self.m_a[1]
                rho_n_t1 = pred_step[:, :, :, 1] * self.std_a[1] + self.m_a[1]
                jx_n_t = prev_state[:, :, :, 3] * self.std_a[3] + self.m_a[3]
                jx_n_t1 = pred_step[:, :, :, 3] * self.std_a[3] + self.m_a[3]
                jy_n_t = prev_state[:, :, :, 5] * self.std_a[5] + self.m_a[5]
                jy_n_t1 = pred_step[:, :, :, 5] * self.std_a[5] + self.m_a[5]
                rho_p_t = prev_state[:, :, :, 0] * self.std_a[0] + self.m_a[0]
                rho_p_t1 = pred_step[:, :, :, 0] * self.std_a[0] + self.m_a[0]
                jx_p_t = prev_state[:, :, :, 2] * self.std_a[2] + self.m_a[2]
                jx_p_t1 = pred_step[:, :, :, 2] * self.std_a[2] + self.m_a[2]
                jy_p_t = prev_state[:, :, :, 4] * self.std_a[4] + self.m_a[4]
                jy_p_t1 = pred_step[:, :, :, 4] * self.std_a[4] + self.m_a[4]
                loss_physics = 0.0
                if self.alpha > 0.0:
                    c_grid = 2.09545
                    d_rho_n = rho_n_t1 - rho_n_t
                    djx_n_dx = (-jx_n_t1[:, 4:, 2:-2] + 8 * jx_n_t1[:, 3:-1, 2:-2] - 8 * jx_n_t1[:, 1:-3, 2:-2] + jx_n_t1[:, :-4, 2:-2]) / 12.0
                    djy_n_dy = (-jy_n_t1[:, 2:-2, 4:] + 8 * jy_n_t1[:, 2:-2, 3:-1] - 8 * jy_n_t1[:, 2:-2, 1:-3] + jy_n_t1[:, 2:-2, :-4]) / 12.0
                    div_J_n = djx_n_dx + djy_n_dy
                    loss_n = F.mse_loss(d_rho_n[:, 2:-2, 2:-2], -c_grid * div_J_n)
                    d_rho_p = rho_p_t1 - rho_p_t
                    djx_p_dx = (-jx_p_t1[:, 4:, 2:-2] + 8 * jx_p_t1[:, 3:-1, 2:-2] - 8 * jx_p_t1[:, 1:-3, 2:-2] + jx_p_t1[:, :-4, 2:-2]) / 12.0
                    djy_p_dy = (-jy_p_t1[:, 2:-2, 4:] + 8 * jy_p_t1[:, 2:-2, 3:-1] - 8 * jy_p_t1[:, 2:-2, 1:-3] + jy_p_t1[:, 2:-2, :-4]) / 12.0
                    div_J_p = djx_p_dx + djy_p_dy
                    loss_p = F.mse_loss(d_rho_p[:, 2:-2, 2:-2], -c_grid * div_J_p)
                    loss_physics = loss_physics + self.alpha * (loss_n + loss_p)
                if self.gamma > 0.0:
                    mass_n_t = torch.sum(rho_n_t, dim=(1, 2))
                    mass_n_t1 = torch.sum(rho_n_t1, dim=(1, 2))
                    mass_p_t = torch.sum(rho_p_t, dim=(1, 2))
                    mass_p_t1 = torch.sum(rho_p_t1, dim=(1, 2))
                    loss_mass = F.mse_loss(mass_n_t1, mass_n_t) + F.mse_loss(mass_p_t1, mass_p_t)
                    loss_physics = loss_physics + self.gamma * loss_mass
                if self.delta > 0.0:
                    px_n_t = torch.sum(jx_n_t, dim=(1, 2))
                    px_n_t1 = torch.sum(jx_n_t1, dim=(1, 2))
                    py_n_t = torch.sum(jy_n_t, dim=(1, 2))
                    py_n_t1 = torch.sum(jy_n_t1, dim=(1, 2))
                    px_p_t = torch.sum(jx_p_t, dim=(1, 2))
                    px_p_t1 = torch.sum(jx_p_t1, dim=(1, 2))
                    py_p_t = torch.sum(jy_p_t, dim=(1, 2))
                    py_p_t1 = torch.sum(jy_p_t1, dim=(1, 2))
                    loss_mom = F.mse_loss(px_n_t1, px_n_t) + F.mse_loss(py_n_t1, py_n_t) + F.mse_loss(px_p_t1, px_p_t) + F.mse_loss(py_p_t1, py_p_t)
                    loss_physics = loss_physics + self.delta * loss_mom
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
                    loss_physics = loss_physics + self.omega * loss_wass
                step_loss = step_loss + loss_physics
            loss += step_loss
        return loss / iteration

class UNOT_Lightning(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(vars(cfg))
        self.cfg = cfg
        self.model = UNO_demo(8, cfg.width, dropout_rate=getattr(cfg, 'dropout_rate', 0.0))
        self.criterion = PhysicsInformedLoss(cfg)

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        max_iter = self.cfg.iteration
        if max_iter > 1:
            phase_length = max(1, self.cfg.num_epochs // max_iter)
            self.current_iteration = min(max_iter, 1 + self.current_epoch // phase_length)
        else:
            self.current_iteration = 1
        if self.current_epoch % phase_length == 0 or self.current_epoch == 0:
            print(f' Epoch {self.current_epoch}: Training with {self.current_iteration} autoregressive iterations!')

    def _iterative_process(self, x, num_iterations):
        batch_size = x.shape[0]
        preds = []
        curr_x = x
        for _ in range(num_iterations):
            im_r = self.model(curr_x)
            if getattr(self.cfg, 'predict_residual', False):
                im_r = curr_x + im_r
            preds.append(im_r)
            curr_x = im_r
        return torch.stack(preds, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]
        num_iters = getattr(self, 'current_iteration', 1)
        pred = self._iterative_process(x, num_iters)
        loss = self.criterion(x, pred, y[:, :num_iters])
        self.log('train_loss', loss / batch_size, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]
        pred = self._iterative_process(x, self.cfg.iteration)
        loss = self.criterion(x, pred, y)
        self.log('val_loss', loss / batch_size, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def on_load_checkpoint(self, checkpoint):
        if 'state_dict' in checkpoint:
            keys_to_delete = [k for k in checkpoint['state_dict'].keys() if '_grid_cache' in k]
            for k in keys_to_delete:
                del checkpoint['state_dict'][k]

    def on_train_epoch_end(self):
        if hasattr(self, 'manual_scheduler'):
            self.manual_scheduler.step()
            self.log('lr-Adam', self.manual_scheduler.get_last_lr()[0], sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.base_lr, weight_decay=self.cfg.weight_decay, amsgrad=False)
        optimizer.add_param_group({'params': self.criterion.parameters(), 'lr': self.cfg.base_lr})
        self.manual_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.num_epochs, eta_min=getattr(self.cfg, 'min_lr', 1e-05))
        return optimizer

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
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir, filename='model_final', save_top_k=1, monitor='val_loss', mode='min', save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
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
    trainer = Trainer(max_epochs=cfg.num_epochs, max_time='00:19:55:00', accelerator='gpu' if cfg.use_gpu else 'cpu', devices=devices, num_nodes=num_nodes, strategy=strategy, callbacks=[checkpoint_callback, lr_monitor], logger=csv_logger, limit_train_batches=limit_train, limit_val_batches=limit_val, use_distributed_sampler=False, fast_dev_run=False)
    last_ckpt_path = os.path.join(args.checkpoint_dir, 'last.ckpt')
    if getattr(cfg, 'load_model', False) and os.path.exists(last_ckpt_path):
        print(f'Resuming training from {last_ckpt_path}')
        trainer.fit(model, train_loader, valid_loader, ckpt_path=last_ckpt_path)
    else:
        trainer.fit(model, train_loader, valid_loader)
if __name__ == '__main__':
    main()