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
from UNO_multi_GPU.physics_dataset import data_prepare
from torch.utils.data import TensorDataset, DataLoader
from Transformer_UNO_v7 import Transformer_UNO_v7
from Transformer_UNO_v7_cond import Transformer_UNO_v7_GRU

torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", category=UserWarning)

class TrainConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class UNOT_V7_Lightning(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(vars(cfg))
        self.cfg = cfg
        
        # in_channels=8 because we pass x [B,H,W,6] + grid [B,H,W,2] inside forward
        if getattr(cfg, 'model_type', 'phys') == 'gru':
            self.model = Transformer_UNO_v7_GRU(
                8, 
                cfg.width,
                dropout_rate=getattr(cfg, 'dropout', 0.0),
                n_cond=1
            )
        else:
            self.model = Transformer_UNO_v7(
                8, 
                cfg.width,
                dropout_rate=getattr(cfg, 'dropout', 0.0)
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
        
        # We decoupled the weight decay into 2 param groups. The old checkpoint has 1.
        # This causes a PyTorch loading crash. We drop the optimizer state to reset Adam.
        if "optimizer_states" in checkpoint:
            checkpoint["optimizer_states"] = []

    def forward(self, x, t_map):
        return self.model(x, t_map)


    def compute_pinn_losses(self, x_initial, current_x, pred_t, y_true, d_rho_p_dt, d_rho_n_dt, div_J_p, div_J_n):
        B, H, W, C = current_x.shape
        alpha = getattr(self.cfg, 'alpha', 1.0)
        gamma = getattr(self.cfg, 'gamma', 0.0)
        delta = getattr(self.cfg, 'delta', 0.0)
        omega = getattr(self.cfg, 'omega', 0.0)
        
        # Exact Temporal Derivative per frame (since t_map is in units of frames)
        # The base scaling factor is (hbar/m*) * (dt / dx) for a single frame.
        dt_per_frame = getattr(self.cfg, 'dt_per_frame', 9.0)  # Physical time per frame (fm/c)
        dx = getattr(self.cfg, 'dx', 0.9)                       # Physical grid spacing (fm)
        hbar_m = getattr(self.cfg, 'hbar_m', 0.2144)            # Kinematic probability current scaling factor
        c_grid = hbar_m * (dt_per_frame / dx)                   # = 2.144 for TDHF90_NORMALIZED
        
        loss_p = F.mse_loss(d_rho_p_dt, -c_grid * div_J_p)
        loss_n = F.mse_loss(d_rho_n_dt, -c_grid * div_J_n)
        loss_cont = loss_p + loss_n
        loss_physics = alpha * loss_cont
        
        loss_mass = torch.tensor(0.0, device=current_x.device)
        loss_mom = torch.tensor(0.0, device=current_x.device)
        loss_swd = torch.tensor(0.0, device=current_x.device)
        
        rho_p = pred_t[..., 0] * self.std_a[0] + self.m_a[0]
        rho_n = pred_t[..., 1] * self.std_a[1] + self.m_a[1]
        
        # 2. Global Mass Conservation (gamma)
        if gamma > 0.0:
            rho_p_0 = x_initial[..., 0] * self.std_a[0] + self.m_a[0]
            rho_n_0 = x_initial[..., 1] * self.std_a[1] + self.m_a[1]
            
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
            
            jx_p_0 = x_initial[..., 4] * self.std_a[4] + self.m_a[4]
            jx_n_0 = x_initial[..., 5] * self.std_a[5] + self.m_a[5]
            jy_p_0 = x_initial[..., 2] * self.std_a[2] + self.m_a[2]
            jy_n_0 = x_initial[..., 3] * self.std_a[3] + self.m_a[3]
            
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

        return loss_physics, loss_cont, loss_mass, loss_mom, loss_swd


    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        B, SeqLen, H, W, C = y.shape
        
        # --- DYNAMIC ITERATION ---
        # Gradually increase autoregressive difficulty
        if self.current_epoch < 100:
            dyn_iter = 1
        elif self.current_epoch < 200:
            dyn_iter = 2
        elif self.current_epoch < 300:
            dyn_iter = 4
        else:
            dyn_iter = getattr(self.cfg, 'iteration', 6)
            
        dyn_iter = min(dyn_iter, SeqLen)
        y = y[:, :dyn_iter]
        SeqLen = dyn_iter
        # -------------------------
        
        device = x.device
        
        # --- TIME-REVERSAL (T-SYMMETRY) AUGMENTATION ---
        # With 50% probability, reverse the arrow of time!
        if torch.rand(1).item() < 0.5:
            # We must flip the ENTIRE sequence
            x_temp = x.clone()
            x = y[:, -1].clone()
            
            # y_true sequence becomes reversed, ending with x_temp
            y_reversed = torch.zeros_like(y)
            for i in range(SeqLen - 1):
                y_reversed[:, i] = y[:, SeqLen - 2 - i]
            y_reversed[:, SeqLen - 1] = x_temp
            y = y_reversed
            
            # Flip the signs of the momentum currents (channels 2, 3, 4, 5)
            # Mathematically for normalized data: J_new = -J_old - 2*(mu/sigma)
            mu_over_sigma = (self.m_a[2:6] / self.std_a[2:6]).to(device).view(1, 1, 1, 4)
            x[..., 2:6] = -x[..., 2:6] - 2 * mu_over_sigma
            y[..., 2:6] = -y[..., 2:6] - 2 * mu_over_sigma
        # -----------------------------------------------
        
        pred_step_val = getattr(self.cfg, 'pred_step', 1)
        t_map = torch.full((B, H, W, 1), float(pred_step_val), device=device).clone().detach()
        
        x_initial = x.clone()
        current_x = x
        total_loss_data = 0
        total_loss_physics = 0
        total_loss_physics_comps = [0.0, 0.0, 0.0, 0.0]
        
        h_gru = None
        is_gru = getattr(self.cfg, 'model_type', 'phys') == 'gru'
        
        for step in range(SeqLen):
            y_true_step = y[:, step]
            
            # 1. Forward pass (Autoregressive step)
            if is_gru:
                pred_t, h_gru = self.model(current_x, t_map, h_gru)
            else:
                pred_t = self.model(current_x, t_map)
            
            # 2. Ground Truth Data Loss
            loss_data = F.mse_loss(pred_t, y_true_step)
            total_loss_data += loss_data
            
            # 3. Physics Loss via Autograd Exact Temporal Derivatives
            rho_p = pred_t[..., 0] * self.std_a[0] + self.m_a[0]
            rho_n = pred_t[..., 1] * self.std_a[1] + self.m_a[1]
            jx_p = pred_t[..., 4] * self.std_a[4] + self.m_a[4]
            jx_n = pred_t[..., 5] * self.std_a[5] + self.m_a[5]
            jy_p = pred_t[..., 2] * self.std_a[2] + self.m_a[2]
            jy_n = pred_t[..., 3] * self.std_a[3] + self.m_a[3]
            
            # Initial State for this step
            rho_p_0 = current_x[..., 0] * self.std_a[0] + self.m_a[0]
            rho_n_0 = current_x[..., 1] * self.std_a[1] + self.m_a[1]
            jx_p_0 = current_x[..., 4] * self.std_a[4] + self.m_a[4]
            jx_n_0 = current_x[..., 5] * self.std_a[5] + self.m_a[5]
            jy_p_0 = current_x[..., 2] * self.std_a[2] + self.m_a[2]
            jy_n_0 = current_x[..., 3] * self.std_a[3] + self.m_a[3]
            
            # Temporal Derivative (Discrete)
            t_scalar = t_map[:, 0, 0, 0].view(B, 1, 1)
            d_rho_p_dt = (rho_p - rho_p_0) / t_scalar
            d_rho_n_dt = (rho_n - rho_n_0) / t_scalar
            
            # Spatial Derivatives at t=0 using Circular Padding
            jx_p_0_pad = F.pad(jx_p_0, pad=(2, 2, 0, 0), mode='circular')
            jy_p_0_pad = F.pad(jy_p_0, pad=(0, 0, 2, 2), mode='circular')
            jx_n_0_pad = F.pad(jx_n_0, pad=(2, 2, 0, 0), mode='circular')
            jy_n_0_pad = F.pad(jy_n_0, pad=(0, 0, 2, 2), mode='circular')
            
            djx_p_0_dx = (-jx_p_0_pad[:, :, 4:] + 8*jx_p_0_pad[:, :, 3:-1] - 8*jx_p_0_pad[:, :, 1:-3] + jx_p_0_pad[:, :, :-4]) / 12.0
            djy_p_0_dy = (-jy_p_0_pad[:, 4:, :] + 8*jy_p_0_pad[:, 3:-1, :] - 8*jy_p_0_pad[:, 1:-3, :] + jy_p_0_pad[:, :-4, :]) / 12.0
            div_J_p_0 = djx_p_0_dx + djy_p_0_dy
            
            djx_n_0_dx = (-jx_n_0_pad[:, :, 4:] + 8*jx_n_0_pad[:, :, 3:-1] - 8*jx_n_0_pad[:, :, 1:-3] + jx_n_0_pad[:, :, :-4]) / 12.0
            djy_n_0_dy = (-jy_n_0_pad[:, 4:, :] + 8*jy_n_0_pad[:, 3:-1, :] - 8*jy_n_0_pad[:, 1:-3, :] + jy_n_0_pad[:, :-4, :]) / 12.0
            div_J_n_0 = djx_n_0_dx + djy_n_0_dy
            
            # Spatial Derivatives at t_target using Circular Padding
            jx_p_pad = F.pad(jx_p, pad=(2, 2, 0, 0), mode='circular')
            jy_p_pad = F.pad(jy_p, pad=(0, 0, 2, 2), mode='circular')
            jx_n_pad = F.pad(jx_n, pad=(2, 2, 0, 0), mode='circular')
            jy_n_pad = F.pad(jy_n, pad=(0, 0, 2, 2), mode='circular')
            
            djx_p_dx = (-jx_p_pad[:, :, 4:] + 8*jx_p_pad[:, :, 3:-1] - 8*jx_p_pad[:, :, 1:-3] + jx_p_pad[:, :, :-4]) / 12.0
            djy_p_dy = (-jy_p_pad[:, 4:, :] + 8*jy_p_pad[:, 3:-1, :] - 8*jy_p_pad[:, 1:-3, :] + jy_p_pad[:, :-4, :]) / 12.0
            div_J_p_t = djx_p_dx + djy_p_dy
            
            djx_n_dx = (-jx_n_pad[:, :, 4:] + 8*jx_n_pad[:, :, 3:-1] - 8*jx_n_pad[:, :, 1:-3] + jx_n_pad[:, :, :-4]) / 12.0
            djy_n_dy = (-jy_n_pad[:, 4:, :] + 8*jy_n_pad[:, 3:-1, :] - 8*jy_n_pad[:, 1:-3, :] + jy_n_pad[:, :-4, :]) / 12.0
            div_J_n_t = djx_n_dx + djy_n_dy
            
            # Midpoint Average is ALWAYS valid here since time step is exactly 1
            div_J_p = (div_J_p_0 + div_J_p_t) / 2.0
            div_J_n = (div_J_n_0 + div_J_n_t) / 2.0
            
            loss_physics_step, loss_c, loss_m, loss_mo, loss_s = self.compute_pinn_losses(x_initial, current_x, pred_t, y_true_step, d_rho_p_dt, d_rho_n_dt, div_J_p, div_J_n)
            total_loss_physics += loss_physics_step.mean()
            total_loss_physics_comps[0] += loss_c.mean()
            total_loss_physics_comps[1] += loss_m.mean()
            total_loss_physics_comps[2] += loss_mo.mean()
            
            # ROLLOUT! Feed the prediction into the next step
            current_x = pred_t
            
        # Average losses over the rollout steps
        loss_data = total_loss_data / SeqLen
        loss_physics = total_loss_physics / SeqLen
        
        # Combine Loss
        beta = getattr(self.cfg, 'beta', 1.0)
        loss = (beta * loss_data) + loss_physics
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)
        self.log('loss_data', loss_data, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)
        self.log('loss_physics', loss_physics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)
        self.log('loss_cont', total_loss_physics_comps[0]/SeqLen, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=B)
        self.log('loss_mass', total_loss_physics_comps[1]/SeqLen, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=B)
        self.log('loss_mom', total_loss_physics_comps[2]/SeqLen, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=B)
        
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch[0], batch[1]
            B, SeqLen, H, W, C = y.shape
            device = x.device

            # --- SAME DYNAMIC ITERATION AS TRAINING (for fair val comparison) ---
            if self.current_epoch < 100:
                dyn_iter = 1
            elif self.current_epoch < 200:
                dyn_iter = 2
            elif self.current_epoch < 300:
                dyn_iter = 4
            else:
                dyn_iter = getattr(self.cfg, 'iteration', 6)

            dyn_iter = min(dyn_iter, SeqLen)
            y = y[:, :dyn_iter]
            SeqLen = dyn_iter
            # ---------------------------------------------------------------------
            
            pred_step_val = getattr(self.cfg, 'pred_step', 1)
            t_map = torch.full((B, H, W, 1), float(pred_step_val), device=device).clone().detach()
            
            x_initial = x.clone()
            current_x = x
            total_loss_data = 0
            total_loss_physics = 0
            total_loss_physics_comps_val = [0.0, 0.0, 0.0, 0.0]
            
            h_gru = None
            is_gru = getattr(self.cfg, 'model_type', 'phys') == 'gru'
            
            for step in range(SeqLen):
                y_true_step = y[:, step]
                
                # 1. Forward pass (Autoregressive step)
                if is_gru:
                    pred_t, h_gru = self.model(current_x, t_map, h_gru)
                else:
                    pred_t = self.model(current_x, t_map)
                
                # 2. Ground Truth Data Loss
                loss_data = F.mse_loss(pred_t, y_true_step)
                total_loss_data += loss_data
                
                # 3. Physics Loss via Autograd Exact Temporal Derivatives
                rho_p = pred_t[..., 0] * self.std_a[0] + self.m_a[0]
                rho_n = pred_t[..., 1] * self.std_a[1] + self.m_a[1]
                jx_p = pred_t[..., 4] * self.std_a[4] + self.m_a[4]
                jx_n = pred_t[..., 5] * self.std_a[5] + self.m_a[5]
                jy_p = pred_t[..., 2] * self.std_a[2] + self.m_a[2]
                jy_n = pred_t[..., 3] * self.std_a[3] + self.m_a[3]
                
                # Initial State for this step
                rho_p_0 = current_x[..., 0] * self.std_a[0] + self.m_a[0]
                rho_n_0 = current_x[..., 1] * self.std_a[1] + self.m_a[1]
                jx_p_0 = current_x[..., 4] * self.std_a[4] + self.m_a[4]
                jx_n_0 = current_x[..., 5] * self.std_a[5] + self.m_a[5]
                jy_p_0 = current_x[..., 2] * self.std_a[2] + self.m_a[2]
                jy_n_0 = current_x[..., 3] * self.std_a[3] + self.m_a[3]
                
                # Temporal Derivative (Discrete)
                t_scalar = t_map[:, 0, 0, 0].view(B, 1, 1)
                d_rho_p_dt = (rho_p - rho_p_0) / t_scalar
                d_rho_n_dt = (rho_n - rho_n_0) / t_scalar
                
                # Spatial Derivatives at t=0 using Circular Padding
                jx_p_0_pad = F.pad(jx_p_0, pad=(2, 2, 0, 0), mode='circular')
                jy_p_0_pad = F.pad(jy_p_0, pad=(0, 0, 2, 2), mode='circular')
                jx_n_0_pad = F.pad(jx_n_0, pad=(2, 2, 0, 0), mode='circular')
                jy_n_0_pad = F.pad(jy_n_0, pad=(0, 0, 2, 2), mode='circular')
                
                djx_p_0_dx = (-jx_p_0_pad[:, :, 4:] + 8*jx_p_0_pad[:, :, 3:-1] - 8*jx_p_0_pad[:, :, 1:-3] + jx_p_0_pad[:, :, :-4]) / 12.0
                djy_p_0_dy = (-jy_p_0_pad[:, 4:, :] + 8*jy_p_0_pad[:, 3:-1, :] - 8*jy_p_0_pad[:, 1:-3, :] + jy_p_0_pad[:, :-4, :]) / 12.0
                div_J_p_0 = djx_p_0_dx + djy_p_0_dy
                
                djx_n_0_dx = (-jx_n_0_pad[:, :, 4:] + 8*jx_n_0_pad[:, :, 3:-1] - 8*jx_n_0_pad[:, :, 1:-3] + jx_n_0_pad[:, :, :-4]) / 12.0
                djy_n_0_dy = (-jy_n_0_pad[:, 4:, :] + 8*jy_n_0_pad[:, 3:-1, :] - 8*jy_n_0_pad[:, 1:-3, :] + jy_n_0_pad[:, :-4, :]) / 12.0
                div_J_n_0 = djx_n_0_dx + djy_n_0_dy
                
                # Spatial Derivatives at t_target using Circular Padding
                jx_p_pad = F.pad(jx_p, pad=(2, 2, 0, 0), mode='circular')
                jy_p_pad = F.pad(jy_p, pad=(0, 0, 2, 2), mode='circular')
                jx_n_pad = F.pad(jx_n, pad=(2, 2, 0, 0), mode='circular')
                jy_n_pad = F.pad(jy_n, pad=(0, 0, 2, 2), mode='circular')
                
                djx_p_dx = (-jx_p_pad[:, :, 4:] + 8*jx_p_pad[:, :, 3:-1] - 8*jx_p_pad[:, :, 1:-3] + jx_p_pad[:, :, :-4]) / 12.0
                djy_p_dy = (-jy_p_pad[:, 4:, :] + 8*jy_p_pad[:, 3:-1, :] - 8*jy_p_pad[:, 1:-3, :] + jy_p_pad[:, :-4, :]) / 12.0
                div_J_p_t = djx_p_dx + djy_p_dy
                
                djx_n_dx = (-jx_n_pad[:, :, 4:] + 8*jx_n_pad[:, :, 3:-1] - 8*jx_n_pad[:, :, 1:-3] + jx_n_pad[:, :, :-4]) / 12.0
                djy_n_dy = (-jy_n_pad[:, 4:, :] + 8*jy_n_pad[:, 3:-1, :] - 8*jy_n_pad[:, 1:-3, :] + jy_n_pad[:, :-4, :]) / 12.0
                div_J_n_t = djx_n_dx + djy_n_dy
                
                # Midpoint Average is ALWAYS valid here since time step is exactly 1
                div_J_p = (div_J_p_0 + div_J_p_t) / 2.0
                div_J_n = (div_J_n_0 + div_J_n_t) / 2.0
                
                loss_physics_step, loss_c, loss_m, loss_mo, loss_s = self.compute_pinn_losses(x_initial, current_x, pred_t, y_true_step, d_rho_p_dt, d_rho_n_dt, div_J_p, div_J_n)
                total_loss_physics += loss_physics_step.mean()
                total_loss_physics_comps_val[0] += loss_c.mean()
                total_loss_physics_comps_val[1] += loss_m.mean()
                total_loss_physics_comps_val[2] += loss_mo.mean()
                
                # ROLLOUT! Feed the prediction into the next step
                current_x = pred_t
                
            # Average losses over the rollout steps
            loss_data = total_loss_data / SeqLen
            loss_physics = total_loss_physics / SeqLen
            
            # Combine Loss
            beta = getattr(self.cfg, 'beta', 1.0)
            loss = (beta * loss_data) + loss_physics
            
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)
            self.log('val_loss_data', loss_data, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=B)
            self.log('val_loss_physics', loss_physics, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=B)
            self.log('val_loss_cont', total_loss_physics_comps_val[0]/SeqLen, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=B)
            self.log('val_loss_mass', total_loss_physics_comps_val[1]/SeqLen, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=B)
            self.log('val_loss_mom', total_loss_physics_comps_val[2]/SeqLen, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=B)
            
            return loss

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        optim_groups = [
            {"params": decay, "weight_decay": getattr(self.cfg, 'weight_decay', 0.01)},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=self.cfg.learning_rate
        )
        t0 = getattr(self.cfg, 'lr_T_0', 200)
        warmup_epochs = getattr(self.cfg, 'warmup_epochs', 20)
        eta_min = getattr(self.cfg, 'eta_min', 1e-5)
        
        if warmup_epochs > 0:
            # start_factor=0.1 with lr=1e-4 means starting at 1e-5
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
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
                param_group['weight_decay'] = getattr(self.cfg, 'weight_decay', 0.01)

def main(cfg):
    print("🚀 Starting Training for Continuous-Time Transformer UNO v7...")

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

    model = UNOT_V7_Lightning(cfg)

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
        filename=f'UNO-iter-{getattr(cfg, "iteration", 1)}-{{epoch:03d}}-{{val_loss:.6f}}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    periodic_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(cfg.checkpoint_dir, cfg.model_name),
        filename=f'UNO-iter-{getattr(cfg, "iteration", 1)}-periodic-{{epoch:03d}}',
        every_n_epochs=10,
        save_top_k=-1
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=getattr(cfg, 'patience', 100),
        verbose=True,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = CSVLogger(cfg.log_dir, name=cfg.model_name)

    num_nodes = int(os.environ.get('SLURM_NNODES', 1))
    global_batch_size = cfg.batch_size * (cfg.gpus if torch.cuda.is_available() else 1) * num_nodes
    
    # Calculate the true number of items (frames) per file based on iteration limit
    predstep = getattr(cfg, 'pred_step', 1)
    iteration = getattr(cfg, 'iteration', 6)
    max_start = 140 - (predstep * iteration)
    if max_start <= 0:
        max_start = 1
        
    total_train_frames = len(train_loader.dataset.file_list) * max_start
    total_val_frames = len(val_loader.dataset.file_list) * max_start
    
    limit_train = max(1, int(total_train_frames / global_batch_size))
    limit_val = max(1, int(total_val_frames / global_batch_size))
    
    print(f"🔄 IterableDataset configured for exactly {limit_train} train steps and {limit_val} val steps per epoch.")
    print(f"   (Based on {max_start} frames per file, Global Batch Size: {global_batch_size})")

    trainer = Trainer(
        max_epochs=cfg.epochs,
        num_nodes=num_nodes,
        devices=cfg.gpus if torch.cuda.is_available() else 1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=1.0,
        strategy='ddp',
        precision=cfg.precision if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback, periodic_checkpoint, early_stop_callback, lr_monitor],
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
