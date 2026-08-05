import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from UNO_multi_GPU.integral_operators import OperatorBlock_2D


# ============================================================
# Transformer_UNO_v7_Phys
# Same as Transformer_UNO_v7 but uses 5 physical conditioning
# scalars instead of a single pred_step sinusoidal embedding.
# ============================================================
class Transformer_UNO_v7_Phys(nn.Module):
    def __init__(self, in_width: int, width: int, pad: int = 0, dropout_rate: float = 0.0, n_cond: int = 5):
        super().__init__()
        self.in_width = in_width
        self.width = width
        self.padding = pad

        self.dropout = nn.Dropout(p=dropout_rate)
        self.dropout2d = nn.Dropout2d(p=dropout_rate)

        # Physical Conditioning MLP (replaces sinusoidal time_mlp)
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width * 16),
            nn.GELU(),
            nn.Linear(width * 16, width * 16)
        )

        # Multi-scale time projectors for decoder injection
        self.time_proj_16 = nn.Sequential(nn.SiLU(), nn.Linear(width * 16, width * 16))
        self.time_proj_8  = nn.Sequential(nn.SiLU(), nn.Linear(width * 16, width * 8))
        self.time_proj_4  = nn.Sequential(nn.SiLU(), nn.Linear(width * 16, width * 4))
        self.time_proj_2  = nn.Sequential(nn.SiLU(), nn.Linear(width * 16, width * 2))

        # Lifting layers
        self.fc = nn.Linear(self.in_width, self.width // 2)
        self.fc0 = nn.Linear(self.width // 2, self.width)

        # High-Frequency Shell Blocks (Increased to 28 modes = exact Nyquist for 56x56)
        self.G_in = OperatorBlock_2D(self.width, self.width, 56, 56, 28, 28)
        self.G_out = OperatorBlock_2D(2 * self.width, self.width, 56, 56, 28, 28)

        # Encoder Operator blocks
        self.G0 = OperatorBlock_2D(self.width, 2 * self.width, 32, 32, 14, 14)
        self.G1 = OperatorBlock_2D(2 * self.width, 4 * self.width, 16, 16, 6, 6)
        self.G2 = OperatorBlock_2D(4 * self.width, 8 * self.width, 8, 8, 3, 3)

        # Bottleneck Operator blocks (4x4 spatial resolution)
        self.G3 = OperatorBlock_2D(8 * self.width, 16 * self.width, 4, 4, 2, 2, Normalize=False)
        self.ln3 = nn.GroupNorm(1, 16 * self.width)

        self.G4 = OperatorBlock_2D(16 * self.width, 16 * self.width, 4, 4, 2, 2, Normalize=False)
        self.ln4 = nn.GroupNorm(1, 16 * self.width)

        self.G5 = OperatorBlock_2D(16 * self.width, 16 * self.width, 4, 4, 2, 2, Normalize=False)
        self.ln5 = nn.GroupNorm(1, 16 * self.width)

        self.G6 = OperatorBlock_2D(16 * self.width, 16 * self.width, 4, 4, 2, 2, Normalize=False)
        self.ln6 = nn.GroupNorm(1, 16 * self.width)

        self.G7 = OperatorBlock_2D(16 * self.width, 16 * self.width, 4, 4, 2, 2, Normalize=False)
        self.ln7 = nn.GroupNorm(1, 16 * self.width)

        self.G8 = OperatorBlock_2D(16 * self.width, 16 * self.width, 4, 4, 2, 2, Normalize=False)
        self.ln8 = nn.GroupNorm(1, 16 * self.width)

        # Decoder Operator blocks
        self.G9  = OperatorBlock_2D(16 * self.width, 8 * self.width, 8, 8, 2, 2)
        self.G10 = OperatorBlock_2D(16 * self.width, 4 * self.width, 16, 16, 3, 3)
        self.G11 = OperatorBlock_2D(8 * self.width, 2 * self.width, 32, 32, 6, 6)
        self.G12 = OperatorBlock_2D(4 * self.width, self.width, 56, 56, 14, 14)

        # Final projections
        self.fc1 = nn.Linear(self.width, 2 * self.width)
        self.fc2 = nn.Linear(2 * self.width, 6)

        self.register_buffer("_grid_cache", None)

    def _build_grid(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        gx = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(1, H, 1, 1)
        gy = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, 1, W, 1)
        gx_exp = gx.expand(1, H, W, 1)
        gy_exp = gy.expand(1, H, W, 1)
        grid = torch.cat([gx_exp, gy_exp], dim=-1)
        return grid

    def get_grid(self, B: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache = self._grid_cache
        if cache is None or cache.shape[1] != H or cache.shape[2] != W or cache.dtype != dtype or cache.device != device:
            new_grid = self._build_grid(H, W, device, dtype)
            self._grid_cache = new_grid
            cache = new_grid
        return cache.expand(B, H, W, 2)

    def forward(self, x: torch.Tensor, t_map: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        device = x.device
        dtype = x.dtype

        # 1. Spatial Grid Encoding
        grid = self.get_grid(B, H, W, device, dtype)
        x_in = torch.cat([x, grid], dim=-1)
        del grid, x

        x_fc = F.gelu(self.fc(x_in))
        x_fc0 = F.gelu(self.fc0(x_fc))
        del x_fc

        x_fc0 = x_fc0.permute(0, 3, 1, 2)

        if self.padding and self.padding > 0:
            pad = self.padding
            x_fc0 = F.pad(x_fc0, [0, pad, 0, pad])
        D1, D2 = x_fc0.shape[-2], x_fc0.shape[-1]

        x_in_56 = self.G_in(x_fc0, D1, D2)

        # 2. Downsampling Encoders
        x_c0 = self.G0(x_in_56, D1 // 2, D2 // 2)
        x_c1 = self.G1(x_c0, D1 // 4, D2 // 4)
        x_c2 = self.G2(x_c1, D1 // 8, D2 // 8)

        # 3. Physical Conditioning Embedding (replaces sinusoidal time_mlp)
        cond = t_map[:, 0, 0, :]  # [B, n_cond]
        t_emb = self.cond_mlp(cond)  # [B, 16*width]

        # 4. Bottleneck with ResNet Connections & LayerNorm
        x_c3 = self.ln3(self.G3(x_c2, D1 // 16, D2 // 16))
        x_c4 = self.ln4(self.G4(x_c3, D1 // 16, D2 // 16)) + x_c3
        x_c5 = self.ln5(self.G5(x_c4, D1 // 16, D2 // 16)) + x_c4
        x_c6 = self.ln6(self.G6(x_c5, D1 // 16, D2 // 16)) + x_c5
        x_c7 = self.ln7(self.G7(x_c6, D1 // 16, D2 // 16)) + x_c6
        x_c8 = self.ln8(self.G8(x_c7, D1 // 16, D2 // 16)) + x_c7

        x_c8 = self.dropout2d(x_c8)

        # 5. Multi-Scale Conditioning Injection in Decoder
        # Inject at Bottleneck (4x4)
        t_scale_16 = self.time_proj_16(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_c8 = x_c8 + t_scale_16

        # Upsample 1: 4x4 -> 8x8
        x_c9 = self.G9(x_c8, D1 // 8, D2 // 8)
        t_scale_8 = self.time_proj_8(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_c9 = x_c9 + t_scale_8
        x_c9 = torch.cat([x_c9, x_c2], dim=1)

        # Upsample 2: 8x8 -> 14x14
        x_c10 = self.G10(x_c9, D1 // 4, D2 // 4)
        x_c10 = F.gelu(x_c10)
        t_scale_4 = self.time_proj_4(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_c10 = x_c10 + t_scale_4
        x_c10 = torch.cat([x_c10, x_c1], dim=1)

        # Upsample 3: 14x14 -> 28x28
        x_c11 = self.G11(x_c10, D1 // 2, D2 // 2)
        x_c11 = F.gelu(x_c11)
        t_scale_2 = self.time_proj_2(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_c11 = x_c11 + t_scale_2
        x_c11 = torch.cat([x_c11, x_c0], dim=1)

        # Upsample 4: 28x28 -> 56x56
        x_c12 = self.G12(x_c11, D1, D2)
        x_c12 = F.gelu(x_c12)
        x_c12 = torch.cat([x_c12, x_in_56], dim=1)

        # Final block
        x_out_56 = self.G_out(x_c12, D1, D2)

        if self.padding and self.padding > 0:
            pad = self.padding
            x_out_56 = x_out_56[..., pad:-pad, pad:-pad]

        x_out = x_out_56.permute(0, 2, 3, 1)
        x_fc1 = F.gelu(self.fc1(x_out))
        x_out = self.fc2(x_fc1)

        return x_out


# ============================================================
# Transformer_UNO_v7_GRU
# Same as Transformer_UNO_v7_Phys but adds a GRU hidden state
# at the bottleneck for full trajectory history awareness.
# ============================================================
class Transformer_UNO_v7_GRU(nn.Module):
    def __init__(self, in_width: int, width: int, pad: int = 0, dropout_rate: float = 0.0, n_cond: int = 5):
        super().__init__()
        self.in_width = in_width
        self.width = width
        self.padding = pad
        self.gru_dim = 16 * width

        self.dropout = nn.Dropout(p=dropout_rate)
        self.dropout2d = nn.Dropout2d(p=dropout_rate)

        # Physical Conditioning MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width * 16),
            nn.GELU(),
            nn.Linear(width * 16, width * 16)
        )

        # Multi-scale time projectors for decoder injection
        self.time_proj_16 = nn.Sequential(nn.SiLU(), nn.Linear(width * 16, width * 16))
        self.time_proj_8  = nn.Sequential(nn.SiLU(), nn.Linear(width * 16, width * 8))
        self.time_proj_4  = nn.Sequential(nn.SiLU(), nn.Linear(width * 16, width * 4))
        self.time_proj_2  = nn.Sequential(nn.SiLU(), nn.Linear(width * 16, width * 2))

        # GRU hidden state at bottleneck
        self.gru_cell = nn.GRUCell(16 * width, 16 * width)
        self.gru_gate = nn.Linear(16 * width, 16 * width)

        # Lifting layers
        self.fc = nn.Linear(self.in_width, self.width // 2)
        self.fc0 = nn.Linear(self.width // 2, self.width)

        # High-Frequency Shell Blocks (Increased to 28 modes = exact Nyquist for 56x56)
        self.G_in = OperatorBlock_2D(self.width, self.width, 56, 56, 28, 28)
        self.G_out = OperatorBlock_2D(2 * self.width, self.width, 56, 56, 28, 28)

        # Encoder Operator blocks
        self.G0 = OperatorBlock_2D(self.width, 2 * self.width, 32, 32, 14, 14)
        self.G1 = OperatorBlock_2D(2 * self.width, 4 * self.width, 16, 16, 6, 6)
        self.G2 = OperatorBlock_2D(4 * self.width, 8 * self.width, 8, 8, 3, 3)

        # Bottleneck Operator blocks (4x4 spatial resolution)
        self.G3 = OperatorBlock_2D(8 * self.width, 16 * self.width, 4, 4, 2, 2, Normalize=False)
        self.ln3 = nn.GroupNorm(1, 16 * self.width)

        self.G4 = OperatorBlock_2D(16 * self.width, 16 * self.width, 4, 4, 2, 2, Normalize=False)
        self.ln4 = nn.GroupNorm(1, 16 * self.width)

        self.G5 = OperatorBlock_2D(16 * self.width, 16 * self.width, 4, 4, 2, 2, Normalize=False)
        self.ln5 = nn.GroupNorm(1, 16 * self.width)

        self.G6 = OperatorBlock_2D(16 * self.width, 16 * self.width, 4, 4, 2, 2, Normalize=False)
        self.ln6 = nn.GroupNorm(1, 16 * self.width)

        self.G7 = OperatorBlock_2D(16 * self.width, 16 * self.width, 4, 4, 2, 2, Normalize=False)
        self.ln7 = nn.GroupNorm(1, 16 * self.width)

        self.G8 = OperatorBlock_2D(16 * self.width, 16 * self.width, 4, 4, 2, 2, Normalize=False)
        self.ln8 = nn.GroupNorm(1, 16 * self.width)

        # Decoder Operator blocks
        self.G9  = OperatorBlock_2D(16 * self.width, 8 * self.width, 8, 8, 2, 2)
        self.G10 = OperatorBlock_2D(16 * self.width, 4 * self.width, 16, 16, 3, 3)
        self.G11 = OperatorBlock_2D(8 * self.width, 2 * self.width, 32, 32, 6, 6)
        self.G12 = OperatorBlock_2D(4 * self.width, self.width, 56, 56, 14, 14)

        # Final projections
        self.fc1 = nn.Linear(self.width, 2 * self.width)
        self.fc2 = nn.Linear(2 * self.width, 6)

        self.register_buffer("_grid_cache", None)

    def _build_grid(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        gx = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(1, H, 1, 1)
        gy = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, 1, W, 1)
        gx_exp = gx.expand(1, H, W, 1)
        gy_exp = gy.expand(1, H, W, 1)
        grid = torch.cat([gx_exp, gy_exp], dim=-1)
        return grid

    def get_grid(self, B: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache = self._grid_cache
        if cache is None or cache.shape[1] != H or cache.shape[2] != W or cache.dtype != dtype or cache.device != device:
            new_grid = self._build_grid(H, W, device, dtype)
            self._grid_cache = new_grid
            cache = new_grid
        return cache.expand(B, H, W, 2)

    def forward(self, x: torch.Tensor, t_map: torch.Tensor, h_gru: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W, _ = x.shape
        device = x.device
        dtype = x.dtype

        # 1. Spatial Grid Encoding
        grid = self.get_grid(B, H, W, device, dtype)
        x_in = torch.cat([x, grid], dim=-1)
        del grid, x

        x_fc = F.gelu(self.fc(x_in))
        x_fc0 = F.gelu(self.fc0(x_fc))
        del x_fc

        x_fc0 = x_fc0.permute(0, 3, 1, 2)

        if self.padding and self.padding > 0:
            pad = self.padding
            x_fc0 = F.pad(x_fc0, [0, pad, 0, pad])
        D1, D2 = x_fc0.shape[-2], x_fc0.shape[-1]

        x_in_56 = self.G_in(x_fc0, D1, D2)

        # 2. Downsampling Encoders
        x_c0 = self.G0(x_in_56, D1 // 2, D2 // 2)
        x_c1 = self.G1(x_c0, D1 // 4, D2 // 4)
        x_c2 = self.G2(x_c1, D1 // 8, D2 // 8)

        # 3. Physical Conditioning Embedding
        cond = t_map[:, 0, 0, :]  # [B, n_cond]
        t_emb = self.cond_mlp(cond)  # [B, 16*width]

        # 4. Bottleneck with ResNet Connections & LayerNorm
        x_c3 = self.ln3(self.G3(x_c2, D1 // 16, D2 // 16))
        x_c4 = self.ln4(self.G4(x_c3, D1 // 16, D2 // 16)) + x_c3
        x_c5 = self.ln5(self.G5(x_c4, D1 // 16, D2 // 16)) + x_c4
        x_c6 = self.ln6(self.G6(x_c5, D1 // 16, D2 // 16)) + x_c5
        x_c7 = self.ln7(self.G7(x_c6, D1 // 16, D2 // 16)) + x_c6
        x_c8 = self.ln8(self.G8(x_c7, D1 // 16, D2 // 16)) + x_c7

        x_c8 = self.dropout2d(x_c8)

        # GRU hidden state update at bottleneck
        if h_gru is None:
            h_gru = torch.zeros(B, self.gru_dim, device=x_c8.device, dtype=x_c8.dtype)
        x_c8_pooled = x_c8.mean(dim=[-1, -2])  # [B, 16*width]
        h_gru = self.gru_cell(x_c8_pooled, h_gru)  # [B, 16*width]
        gate = torch.sigmoid(self.gru_gate(h_gru)).unsqueeze(-1).unsqueeze(-1)
        x_c8 = x_c8 * gate  # gated modulation

        # 5. Multi-Scale Conditioning Injection in Decoder
        # Inject at Bottleneck (4x4)
        t_scale_16 = self.time_proj_16(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_c8 = x_c8 + t_scale_16

        # Upsample 1: 4x4 -> 8x8
        x_c9 = self.G9(x_c8, D1 // 8, D2 // 8)
        t_scale_8 = self.time_proj_8(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_c9 = x_c9 + t_scale_8
        x_c9 = torch.cat([x_c9, x_c2], dim=1)

        # Upsample 2: 8x8 -> 14x14
        x_c10 = self.G10(x_c9, D1 // 4, D2 // 4)
        x_c10 = F.gelu(x_c10)
        t_scale_4 = self.time_proj_4(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_c10 = x_c10 + t_scale_4
        x_c10 = torch.cat([x_c10, x_c1], dim=1)

        # Upsample 3: 14x14 -> 28x28
        x_c11 = self.G11(x_c10, D1 // 2, D2 // 2)
        x_c11 = F.gelu(x_c11)
        t_scale_2 = self.time_proj_2(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_c11 = x_c11 + t_scale_2
        x_c11 = torch.cat([x_c11, x_c0], dim=1)

        # Upsample 4: 28x28 -> 56x56
        x_c12 = self.G12(x_c11, D1, D2)
        x_c12 = F.gelu(x_c12)
        x_c12 = torch.cat([x_c12, x_in_56], dim=1)

        # Final block
        x_out_56 = self.G_out(x_c12, D1, D2)

        if self.padding and self.padding > 0:
            pad = self.padding
            x_out_56 = x_out_56[..., pad:-pad, pad:-pad]

        x_out = x_out_56.permute(0, 2, 3, 1)
        x_fc1 = F.gelu(self.fc1(x_out))
        x_out = self.fc2(x_fc1)

        return x_out, h_gru
