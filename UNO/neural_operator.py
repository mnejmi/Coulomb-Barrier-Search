import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from fourier_layers import FourierOperatorBlock2d

class UNetNeuralOperator(nn.Module):

    def __init__(self, in_channels: int, width: int, pad: int=0):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.padding = pad
        self.fc = nn.Linear(self.in_channels, self.width // 2)
        self.fc0 = nn.Linear(self.width // 2, self.width)
        self.G0 = FourierOperatorBlock2d(self.width, 2 * self.width, 32, 32, 14, 14)
        self.G1 = FourierOperatorBlock2d(2 * self.width, 4 * self.width, 16, 16, 6, 6)
        self.G2 = FourierOperatorBlock2d(4 * self.width, 8 * self.width, 8, 8, 3, 3)
        self.G3 = FourierOperatorBlock2d(8 * self.width, 16 * self.width, 4, 4, 2, 2)
        self.G4 = FourierOperatorBlock2d(16 * self.width, 16 * self.width, 4, 4, 2, 2)
        self.G5 = FourierOperatorBlock2d(16 * self.width, 16 * self.width, 4, 4, 2, 2)
        self.G6 = FourierOperatorBlock2d(16 * self.width, 16 * self.width, 4, 4, 2, 2)
        self.G7 = FourierOperatorBlock2d(16 * self.width, 16 * self.width, 4, 4, 2, 2)
        self.G8 = FourierOperatorBlock2d(16 * self.width, 16 * self.width, 4, 4, 2, 2)
        self.G9 = FourierOperatorBlock2d(16 * self.width, 8 * self.width, 8, 8, 2, 2)
        self.G10 = FourierOperatorBlock2d(16 * self.width, 4 * self.width, 16, 16, 3, 3)
        self.G11 = FourierOperatorBlock2d(8 * self.width, 2 * self.width, 32, 32, 6, 6)
        self.G12 = FourierOperatorBlock2d(4 * self.width, self.width, 56, 56, 14, 14)
        self.fc1 = nn.Linear(self.width, 2 * self.width)
        self.fc2 = nn.Linear(2 * self.width, 6)
        self.register_buffer('_grid_cache', None)

    def _build_grid(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        gx = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(1, H, 1, 1)
        gy = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, 1, W, 1)
        gx_exp = gx.expand(1, H, W, 1)
        gy_exp = gy.expand(1, H, W, 1)
        grid = torch.cat([gx_exp, gy_exp], dim=-1)
        return grid

    def get_grid(self, B: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache = self._grid_cache
        if cache is None or cache.shape[1] != H or cache.shape[2] != W or (cache.dtype != dtype) or (cache.device != device):
            new_grid = self._build_grid(H, W, device, dtype)
            self._grid_cache = new_grid
            cache = new_grid
        return cache.expand(B, H, W, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        device = x.device
        dtype = x.dtype
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
        D1, D2 = (x_fc0.shape[-2], x_fc0.shape[-1])
        x_c0 = self.G0(x_fc0, D1 // 2, D2 // 2)
        x_c1 = self.G1(x_c0, D1 // 4, D2 // 4)
        x_c2 = self.G2(x_c1, D1 // 8, D2 // 8)
        x_c3 = self.G3(x_c2, D1 // 16, D2 // 16)
        x_c4 = self.G4(x_c3, D1 // 16, D2 // 16)
        x_c5 = self.G5(x_c4, D1 // 16, D2 // 16)
        x_c6 = self.G6(x_c5, D1 // 16, D2 // 16)
        x_c7 = self.G7(x_c6, D1 // 16, D2 // 16)
        x_c8 = self.G8(x_c7, D1 // 16, D2 // 16)
        x_c9 = self.G9(x_c8, D1 // 8, D2 // 8)
        x_c9 = torch.cat([x_c9, x_c2], dim=1)
        x_c10 = self.G10(x_c9, D1 // 4, D2 // 4)
        x_c10 = torch.cat([x_c10, x_c1], dim=1)
        x_c11 = self.G11(x_c10, D1 // 2, D2 // 2)
        x_c11 = torch.cat([x_c11, x_c0], dim=1)
        x_c12 = self.G12(x_c11, D1, D2)
        if self.padding and self.padding > 0:
            pad = self.padding
            x_c12 = x_c12[..., :-pad, :-pad]
        x_c12 = x_c12.permute(0, 2, 3, 1)
        x_fc1 = F.gelu(self.fc1(x_c12))
        x_out = self.fc2(x_fc1)
        return x_out