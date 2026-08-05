import argparse
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
import yaml
import warnings
from physics_dataset import data_prepare
from torch.utils.data import TensorDataset, DataLoader
from Transformer_UNO_v6 import Transformer_UNO_v6

torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", category=UserWarning)

class TrainConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class UNOT_V6_Lightning(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(vars(cfg))
        self.cfg = cfg
        
        # in_channels=8 because we pass x [B,H,W,6] + grid [B,H,W,2] inside forward
        self.model = Transformer_UNO_v6(
            8, 
            cfg.width,
            dropout_rate=getattr(cfg, 'dropout_rate', 0.0)
        )
        
        # Load Global Normalization Stats for Physics Loss
        stats_path = os.path.join(cfg.data_dir, 'global_normalization_stats.pt')
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location='cpu')
            self.register_buffer('m_a', stats['m'].float())
            self.register_buffer('std_a', stats['std'].float())
        else:
            self.register_buffer('m_a', torch.zeros(6))
            self.register_buffer('std_a', torch.ones(6))

        # Automatic Optimization must be manual if we want to use create_graph=True
        # Wait, Lightning supports create_graph=True in the backward pass inherently 
        # if the loss requires it. However, PyTorch Lightning automatic optimization 
        # might clash with retain_graph. Let's see if automatic works.

    def on_load_checkpoint(self, checkpoint):
        if "model._grid_cache" in checkpoint["state_dict"]:
            del checkpoint["state_dict"]["model._grid_cache"]

    def forward(self, x, t_map):
        return self.model(x, t_map)


    def compute_pinn_losses(self, x, pred_t, y_true, d_rho_p_dt, d_rho_n_dt, div_J_p, div_J_n):
        B, H, W, C = x.shape
        alpha = getattr(self.cfg, 'alpha', 1.0)
        gamma = getattr(self.cfg, 'gamma', 0.0)
        delta = getattr(self.cfg, 'delta', 0.0)
        omega = getattr(self.cfg, 'omega', 0.0)
        
        # Exact Temporal Derivative per frame (since t_map is in units of frames)
        # The base scaling factor is dt / dx for a single frame.
        # dt/dx = 2.09545 (from V5 for 3 steps) / 3.0 = 0.698483
        c_grid = 2.09545 / 3.0
        
        loss_p = F.mse_loss(d_rho_p_dt[:, 2:-2, 2:-2], -c_grid * div_J_p)
        loss_n = F.mse_loss(d_rho_n_dt[:, 2:-2, 2:-2], -c_grid * div_J_n)
        loss_physics = alpha * (loss_p + loss_n)
        
        rho_p = pred_t[..., 0] * self.std_a[0] + self.m_a[0]
        rho_n = pred_t[..., 1] * self.std_a[1] + self.m_a[1]
        
        # 2. Global Mass Conservation (gamma)
        if gamma > 0.0:
            rho_p_0 = x[..., 0] * self.std_a[0] + self.m_a[0]
            rho_n_0 = x[..., 1] * self.std_a[1] + self.m_a[1]
            
            dV = 0.81  # dx * dy = 0.9 * 0.9
            mass_p_0 = torch.sum(rho_p_0, dim=(1,2)) * dV
            mass_n_0 = torch.sum(rho_n_0, dim=(1,2)) * dV
            mass_p_t = torch.sum(rho_p, dim=(1,2)) * dV
            mass_n_t = torch.sum(rho_n, dim=(1,2)) * dV
            
            error_mass_p = (mass_p_t - mass_p_0) / (self.std_a[0] * dV * (H * W))
            error_mass_n = (mass_n_t - mass_n_0) / (self.std_a[1] * dV * (H * W))
            
            loss_mass = F.mse_loss(error_mass_p, torch.zeros_like(error_mass_p)) +                         F.mse_loss(error_mass_n, torch.zeros_like(error_mass_n))
            loss_physics = loss_physics + gamma * loss_mass
            
        # 3. Global Momentum Conservation (delta)
        if delta > 0.0:
            jx_p = pred_t[..., 4] * self.std_a[4] + self.m_a[4]
            jx_n = pred_t[..., 5] * self.std_a[5] + self.m_a[5]
            jy_p = pred_t[..., 2] * self.std_a[2] + self.m_a[2]
            jy_n = pred_t[..., 3] * self.std_a[3] + self.m_a[3]
            
            jx_p_0 = x[..., 4] * self.std_a[4] + self.m_a[4]
            jx_n_0 = x[..., 5] * self.std_a[5] + self.m_a[5]
            jy_p_0 = x[..., 2] * self.std_a[2] + self.m_a[2]
            jy_n_0 = x[..., 3] * self.std_a[3] + self.m_a[3]
            
            px_p_0 = torch.sum(jx_p_0, dim=(1,2))
            px_n_0 = torch.sum(jx_n_0, dim=(1,2))
            py_p_0 = torch.sum(jy_p_0, dim=(1,2))
            py_n_0 = torch.sum(jy_n_0, dim=(1,2))
            
            px_p_t = torch.sum(jx_p, dim=(1,2))
            px_n_t = torch.sum(jx_n, dim=(1,2))
            py_p_t = torch.sum(jy_p, dim=(1,2))
            py_n_t = torch.sum(jy_n, dim=(1,2))
            
            error_px_p = (px_p_t - px_p_0) / (self.std_a[2] * (H * W))
            error_px_n = (px_n_t - px_n_0) / (self.std_a[3] * (H * W))
            error_py_p = (py_p_t - py_p_0) / (self.std_a[4] * (H * W))
            error_py_n = (py_n_t - py_n_0) / (self.std_a[5] * (H * W))
            
            loss_mom = F.mse_loss(error_px_p, torch.zeros_like(error_px_p)) +                        F.mse_loss(error_px_n, torch.zeros_like(error_px_n)) +                        F.mse_loss(error_py_p, torch.zeros_like(error_py_p)) +                        F.mse_loss(error_py_n, torch.zeros_like(error_py_n))
            loss_physics = loss_physics + delta * loss_mom

        # 4. Sliced Wasserstein Distance (omega)
        if omega > 0.0:
            rho_p_true = y_true[..., 0] * self.std_a[0] + self.m_a[0]
            rho_n_true = y_true[..., 1] * self.std_a[1] + self.m_a[1]
            
            rho_p_pos = F.relu(rho_p) + 1e-8
            rho_n_pos = F.relu(rho_n) + 1e-8
            rho_p_true_pos = F.relu(rho_p_true) + 1e-8
            rho_n_true_pos = F.relu(rho_n_true) + 1e-8
            
            marg_x_p = torch.sum(rho_p_pos, dim=1)
            marg_x_n = torch.sum(rho_n_pos, dim=1)
            marg_x_p_true = torch.sum(rho_p_true_pos, dim=1)
            marg_x_n_true = torch.sum(rho_n_true_pos, dim=1)
            
            marg_y_p = torch.sum(rho_p_pos, dim=2)
            marg_y_n = torch.sum(rho_n_pos, dim=2)
            marg_y_p_true = torch.sum(rho_p_true_pos, dim=2)
            marg_y_n_true = torch.sum(rho_n_true_pos, dim=2)
            
            cdf_x_p = torch.cumsum(marg_x_p, dim=1)
            cdf_x_n = torch.cumsum(marg_x_n, dim=1)
            cdf_x_p_true = torch.cumsum(marg_x_p_true, dim=1)
            cdf_x_n_true = torch.cumsum(marg_x_n_true, dim=1)
            
            cdf_y_p = torch.cumsum(marg_y_p, dim=1)
            cdf_y_n = torch.cumsum(marg_y_n, dim=1)
            cdf_y_p_true = torch.cumsum(marg_y_p_true, dim=1)
            cdf_y_n_true = torch.cumsum(marg_y_n_true, dim=1)
            
            cdf_x_p = cdf_x_p / cdf_x_p[:, -1:]
            cdf_x_n = cdf_x_n / cdf_x_n[:, -1:]
            cdf_x_p_true = cdf_x_p_true / cdf_x_p_true[:, -1:]
            cdf_x_n_true = cdf_x_n_true / cdf_x_n_true[:, -1:]
            
            cdf_y_p = cdf_y_p / cdf_y_p[:, -1:]
            cdf_y_n = cdf_y_n / cdf_y_n[:, -1:]
            cdf_y_p_true = cdf_y_p_true / cdf_y_p_true[:, -1:]
            cdf_y_n_true = cdf_y_n_true / cdf_y_n_true[:, -1:]
            
            loss_swd = F.mse_loss(cdf_x_p, cdf_x_p_true) + F.mse_loss(cdf_x_n, cdf_x_n_true) +                        F.mse_loss(cdf_y_p, cdf_y_p_true) + F.mse_loss(cdf_y_n, cdf_y_n_true)
            loss_physics = loss_physics + omega * loss_swd

        return loss_physics


    def training_step(self, batch, batch_idx):
        x, y = batch
        B, SeqLen, H, W, C = y.shape
        device = x.device
        
        # 1. Randomly sample a target index from 0 to SeqLen - 1
        t_idx_target = torch.randint(0, SeqLen, (B, 1), device=device)
        t_idx_long = t_idx_target.long().squeeze(1)
        y_true = y[torch.arange(B), t_idx_long]
        
        # --- TIME-REVERSAL (T-SYMMETRY) AUGMENTATION ---
        # With 50% probability, reverse the arrow of time!
        if torch.rand(1).item() < 0.5:
            # Swap x (past) and y_true (future)
            x_temp = x.clone()
            x = y_true
            y_true = x_temp
            
            # Flip the signs of the momentum currents (channels 2, 3, 4, 5)
            # Mathematically for normalized data: J_new = -J_old - 2*(mu/sigma)
            mu_over_sigma = (self.m_a[2:6] / self.std_a[2:6]).to(device).view(1, 1, 1, 4)
            x[..., 2:6] = -x[..., 2:6] - 2 * mu_over_sigma
            y_true[..., 2:6] = -y_true[..., 2:6] - 2 * mu_over_sigma
        # -----------------------------------------------
        
        # Continuous time map: t_map = exact number of frames into the future
        pred_step_val = getattr(self.cfg, 'pred_step', 3)
        t_map = ((t_idx_target + 1).float() * pred_step_val).view(B, 1, 1, 1).expand(B, H, W, 1).clone().detach().requires_grad_(True)
        
        # 2. Forward pass
        pred_t = self.model(x, t_map)
        
        # 3. Ground Truth Data Loss
        loss_data = F.mse_loss(pred_t, y_true)
        
        # 4. Physics Loss via Autograd Exact Temporal Derivatives
        rho_p = pred_t[..., 0] * self.std_a[0] + self.m_a[0]
        rho_n = pred_t[..., 1] * self.std_a[1] + self.m_a[1]
        jx_p = pred_t[..., 4] * self.std_a[4] + self.m_a[4]
        jx_n = pred_t[..., 5] * self.std_a[5] + self.m_a[5]
        jy_p = pred_t[..., 2] * self.std_a[2] + self.m_a[2]
        jy_n = pred_t[..., 3] * self.std_a[3] + self.m_a[3]
        
        # ---------------------------------------------------------
        # EXACT DISCRETE MIDPOINT RULE (User Rule 8 & Blur Fix)
        # ---------------------------------------------------------
        # Initial State (t = 0)
        rho_p_0 = x[..., 0] * self.std_a[0] + self.m_a[0]
        rho_n_0 = x[..., 1] * self.std_a[1] + self.m_a[1]
        jx_p_0 = x[..., 4] * self.std_a[4] + self.m_a[4]
        jx_n_0 = x[..., 5] * self.std_a[5] + self.m_a[5]
        jy_p_0 = x[..., 2] * self.std_a[2] + self.m_a[2]
        jy_n_0 = x[..., 3] * self.std_a[3] + self.m_a[3]
        
        # Temporal Derivative (Discrete)
        t_scalar = t_map[:, 0, 0, 0].view(B, 1, 1)
        d_rho_p_dt = (rho_p - rho_p_0) / t_scalar
        d_rho_n_dt = (rho_n - rho_n_0) / t_scalar
        
        # Spatial Derivatives at t=0
        # X derivative: slice along W (dim 2)
        djx_p_0_dx = (-jx_p_0[:, 2:-2, 4:] + 8*jx_p_0[:, 2:-2, 3:-1] - 8*jx_p_0[:, 2:-2, 1:-3] + jx_p_0[:, 2:-2, :-4]) / 12.0
        # Y derivative: slice along H (dim 1)
        djy_p_0_dy = (-jy_p_0[:, 4:, 2:-2] + 8*jy_p_0[:, 3:-1, 2:-2] - 8*jy_p_0[:, 1:-3, 2:-2] + jy_p_0[:, :-4, 2:-2]) / 12.0
        div_J_p_0 = djx_p_0_dx + djy_p_0_dy
        
        djx_n_0_dx = (-jx_n_0[:, 2:-2, 4:] + 8*jx_n_0[:, 2:-2, 3:-1] - 8*jx_n_0[:, 2:-2, 1:-3] + jx_n_0[:, 2:-2, :-4]) / 12.0
        djy_n_0_dy = (-jy_n_0[:, 4:, 2:-2] + 8*jy_n_0[:, 3:-1, 2:-2] - 8*jy_n_0[:, 1:-3, 2:-2] + jy_n_0[:, :-4, 2:-2]) / 12.0
        div_J_n_0 = djx_n_0_dx + djy_n_0_dy
        
        # Spatial Derivatives at t_target
        djx_p_dx = (-jx_p[:, 2:-2, 4:] + 8*jx_p[:, 2:-2, 3:-1] - 8*jx_p[:, 2:-2, 1:-3] + jx_p[:, 2:-2, :-4]) / 12.0
        djy_p_dy = (-jy_p[:, 4:, 2:-2] + 8*jy_p[:, 3:-1, 2:-2] - 8*jy_p[:, 1:-3, 2:-2] + jy_p[:, :-4, 2:-2]) / 12.0
        div_J_p_t = djx_p_dx + djy_p_dy
        
        djx_n_dx = (-jx_n[:, 2:-2, 4:] + 8*jx_n[:, 2:-2, 3:-1] - 8*jx_n[:, 2:-2, 1:-3] + jx_n[:, 2:-2, :-4]) / 12.0
        djy_n_dy = (-jy_n[:, 4:, 2:-2] + 8*jy_n[:, 3:-1, 2:-2] - 8*jy_n[:, 1:-3, 2:-2] + jy_n[:, :-4, 2:-2]) / 12.0
        div_J_n_t = djx_n_dx + djy_n_dy
        
        # Midpoint Average!
        div_J_p = (div_J_p_0 + div_J_p_t) / 2.0
        div_J_n = (div_J_n_0 + div_J_n_t) / 2.0
        
        loss_physics = self.compute_pinn_losses(x, pred_t, y_true, d_rho_p_dt, d_rho_n_dt, div_J_p, div_J_n)
        
        # Combine Loss
        beta = getattr(self.cfg, 'beta', 1.0)
        loss = (beta * loss_data) + loss_physics
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)
        self.log('loss_data', loss_data, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)
        self.log('loss_physics', loss_physics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)
        
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            x, y = batch
            B, SeqLen, H, W, C = y.shape
            device = x.device
            
            # 1. Randomly sample a target index from 0 to SeqLen - 1
            t_idx_target = torch.randint(0, SeqLen, (B, 1), device=device)
            x = x.view(B, H, W, -1)
            
            # Continuous time map: t_map = exact number of frames into the future
            pred_step_val = getattr(self.cfg, 'pred_step', 3)
            t_map = ((t_idx_target + 1).float() * pred_step_val).view(B, 1, 1, 1).expand(B, H, W, 1).clone().detach().requires_grad_(True)
            
            # 2. Forward pass
            pred_t = self.model(x, t_map)
            
            # 3. Ground Truth Data Loss
            t_idx_long = t_idx_target.long().squeeze(1)
            y_true = y[torch.arange(B), t_idx_long]
            loss_data = F.mse_loss(pred_t, y_true)
            
            # 4. Physics Loss via Autograd Exact Temporal Derivatives
            rho_p = pred_t[..., 0] * self.std_a[0] + self.m_a[0]
            rho_n = pred_t[..., 1] * self.std_a[1] + self.m_a[1]
            jx_p = pred_t[..., 4] * self.std_a[4] + self.m_a[4]
            jx_n = pred_t[..., 5] * self.std_a[5] + self.m_a[5]
            jy_p = pred_t[..., 2] * self.std_a[2] + self.m_a[2]
            jy_n = pred_t[..., 3] * self.std_a[3] + self.m_a[3]
            
            # ---------------------------------------------------------
            # EXACT DISCRETE MIDPOINT RULE
            # ---------------------------------------------------------
            rho_p_0 = x[..., 0] * self.std_a[0] + self.m_a[0]
            rho_n_0 = x[..., 1] * self.std_a[1] + self.m_a[1]
            jx_p_0 = x[..., 4] * self.std_a[4] + self.m_a[4]
            jx_n_0 = x[..., 5] * self.std_a[5] + self.m_a[5]
            jy_p_0 = x[..., 2] * self.std_a[2] + self.m_a[2]
            jy_n_0 = x[..., 3] * self.std_a[3] + self.m_a[3]
            
            t_scalar = t_map[:, 0, 0, 0].view(B, 1, 1)
            d_rho_p_dt = (rho_p - rho_p_0) / t_scalar
            d_rho_n_dt = (rho_n - rho_n_0) / t_scalar
            
            djx_p_0_dx = (-jx_p_0[:, 4:, 2:-2] + 8*jx_p_0[:, 3:-1, 2:-2] - 8*jx_p_0[:, 1:-3, 2:-2] + jx_p_0[:, :-4, 2:-2]) / 12.0
            djy_p_0_dy = (-jy_p_0[:, 2:-2, 4:] + 8*jy_p_0[:, 2:-2, 3:-1] - 8*jy_p_0[:, 2:-2, 1:-3] + jy_p_0[:, 2:-2, :-4]) / 12.0
            div_J_p_0 = djx_p_0_dx + djy_p_0_dy
            
            djx_n_0_dx = (-jx_n_0[:, 2:-2, 4:] + 8*jx_n_0[:, 2:-2, 3:-1] - 8*jx_n_0[:, 2:-2, 1:-3] + jx_n_0[:, 2:-2, :-4]) / 12.0
            djy_n_0_dy = (-jy_n_0[:, 4:, 2:-2] + 8*jy_n_0[:, 3:-1, 2:-2] - 8*jy_n_0[:, 1:-3, 2:-2] + jy_n_0[:, :-4, 2:-2]) / 12.0
            div_J_n_0 = djx_n_0_dx + djy_n_0_dy
            
            djx_p_dx = (-jx_p[:, 2:-2, 4:] + 8*jx_p[:, 2:-2, 3:-1] - 8*jx_p[:, 2:-2, 1:-3] + jx_p[:, 2:-2, :-4]) / 12.0
            djy_p_dy = (-jy_p[:, 4:, 2:-2] + 8*jy_p[:, 3:-1, 2:-2] - 8*jy_p[:, 1:-3, 2:-2] + jy_p[:, :-4, 2:-2]) / 12.0
            div_J_p_t = djx_p_dx + djy_p_dy
            
            djx_n_dx = (-jx_n[:, 2:-2, 4:] + 8*jx_n[:, 2:-2, 3:-1] - 8*jx_n[:, 2:-2, 1:-3] + jx_n[:, 2:-2, :-4]) / 12.0
            djy_n_dy = (-jy_n[:, 4:, 2:-2] + 8*jy_n[:, 3:-1, 2:-2] - 8*jy_n[:, 1:-3, 2:-2] + jy_n[:, :-4, 2:-2]) / 12.0
            div_J_n_t = djx_n_dx + djy_n_dy
            
            div_J_p = (div_J_p_0 + div_J_p_t) / 2.0
            div_J_n = (div_J_n_0 + div_J_n_t) / 2.0
            
            loss_physics = self.compute_pinn_losses(x, pred_t, y_true, d_rho_p_dt, d_rho_n_dt, div_J_p, div_J_n)
            
            # Combine Loss
            beta = getattr(self.cfg, 'beta', 1.0)
            loss = (beta * loss_data) + loss_physics
            
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)
            self.log('val_loss_data', loss_data, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)
            self.log('val_loss_physics', loss_physics, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)
            
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.cfg.learning_rate, 
            weight_decay=self.cfg.weight_decay
        )
        t0 = getattr(self.cfg, 'lr_T_0', 200)
        warmup_epochs = getattr(self.cfg, 'warmup_epochs', 0)
        eta_min = getattr(self.cfg, 'eta_min', 1e-6)
        
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=t0 - warmup_epochs, 
                T_mult=1, 
                eta_min=eta_min
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=t0, 
                T_mult=1, 
                eta_min=eta_min
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_start(self):
        schedulers = self.trainer.lr_scheduler_configs
        if schedulers:
            s = schedulers[0].scheduler
            if getattr(s, 'last_epoch', 0) < self.current_epoch:
                s.step(self.current_epoch)
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group['lr'] = s.get_last_lr()[0]

def main(cfg):
    print("🚀 Starting Training for Continuous-Time Transformer UNO v6...")

    if getattr(cfg, 'cpu_per_task', None) is not None:
        optimal_workers = max(1, int(cfg.cpu_per_task) - 1)
    elif 'SLURM_CPUS_PER_TASK' in os.environ:
        optimal_workers = max(1, int(os.environ['SLURM_CPUS_PER_TASK']) - 1)
    else:
        optimal_workers = getattr(cfg, 'num_workers', 1)

    print(f"🚀 Dynamically configured DataLoader with num_workers = {optimal_workers}")

    # Use cfg.data_path if available (e.g. copied to local SSD), otherwise use cfg.data_dir
    inputpath = getattr(cfg, 'data_path', cfg.data_dir)
    train_dataset, valid_dataset = data_prepare(cfg, inputpath)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        pin_memory=True, 
        num_workers=optimal_workers
    )
    val_loader = DataLoader(
        valid_dataset, 
        batch_size=cfg.batch_size, 
        pin_memory=True, 
        num_workers=optimal_workers
    )

    model = UNOT_V6_Lightning(cfg)

    # Load from Checkpoint?
    checkpoint_path = None
    if getattr(cfg, 'load_model', False):
        import glob
        ckpt_dir = os.path.join(cfg.checkpoint_dir, cfg.model_name)
        if os.path.exists(ckpt_dir):
            checkpoints = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
            if checkpoints:
                import re
                def ckpt_version_key(path):
                    name = os.path.basename(path)
                    m = re.search(r'last(?:-v(\d+))?\.ckpt', name)
                    return int(m.group(1) or 0) if m else -1
                last_ckpts = [p for p in checkpoints if 'last' in os.path.basename(p)]
                if last_ckpts:
                    last_ckpts.sort(key=ckpt_version_key)
                    checkpoint_path = last_ckpts[-1]
                else:
                    checkpoints.sort(key=os.path.getmtime)
                    checkpoint_path = checkpoints[-1]
                print(f"🔄 Resuming from checkpoint: {checkpoint_path}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.checkpoint_dir, cfg.model_name),
        filename='UNO-{epoch:03d}-{val_loss:.6f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=getattr(cfg, 'patience', 100),
        verbose=True,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = CSVLogger(cfg.log_dir, name=cfg.model_name)

    global_batch_size = cfg.batch_size * (cfg.gpus if torch.cuda.is_available() else 1)
    limit_train = max(1, int(len(train_loader.dataset.file_list) / global_batch_size))
    limit_val = max(1, int(len(val_loader.dataset.file_list) / global_batch_size))
    
    print(f"🔄 IterableDataset configured for exactly {limit_train} train steps and {limit_val} val steps per epoch.")

    trainer = Trainer(
        max_epochs=cfg.epochs,
        devices=cfg.gpus if torch.cuda.is_available() else 1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=1.0,
        strategy='ddp_find_unused_parameters_true' if (torch.cuda.is_available() and cfg.gpus > 1) else 'auto',
        precision=cfg.precision if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_transformer_v6.yaml', help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    cfg = TrainConfig(**config_dict)
    
    # Slurm SIGUSR1 Trap logic
    import signal
    def sigusr1_handler(signum, frame):
        print("=====================================")
        print("Caught SIGUSR1! Time limit approaching.")
        print("Requeuing job...")
        with open(args.config, 'r') as f:
            lines = f.readlines()
        with open(args.config, 'w') as f:
            for line in lines:
                if line.strip().startswith('load_model:'):
                    f.write('load_model: True\n')
                else:
                    f.write(line)
        import subprocess
        job_id = os.environ.get('SLURM_JOB_ID')
        if job_id:
            subprocess.run(['scontrol', 'requeue', job_id])
        import sys
        sys.exit(0)

    signal.signal(signal.SIGUSR1, sigusr1_handler)
    
    main(cfg)
