import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
import matplotlib.pyplot as plt
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
from Adam import Adam
from torchsummary import summary
import gc
import math
from fourier_layers import *

class UNetNeuralOperator(nn.Module):

    def __init__(self, in_width, width, pad=0):
        super(UNetNeuralOperator, self).__init__()
        self.in_width = in_width
        self.width = width
        self.padding = pad
        self.fc = nn.Linear(self.in_width, self.width // 2)
        self.fc0 = nn.Linear(self.width // 2, self.width)
        self.G0 = FourierOperatorBlock2d(self.width, 2 * self.width, 64, 64, 28, 28)
        self.G1 = FourierOperatorBlock2d(2 * self.width, 4 * self.width, 32, 32, 12, 12)
        self.G2 = FourierOperatorBlock2d(4 * self.width, 8 * self.width, 16, 16, 6, 6)
        self.G3 = FourierOperatorBlock2d(8 * self.width, 16 * self.width, 8, 8, 3, 3)
        self.G4 = FourierOperatorBlock2d(16 * self.width, 16 * self.width, 8, 8, 3, 3)
        self.G5 = FourierOperatorBlock2d(16 * self.width, 16 * self.width, 8, 8, 3, 3)
        self.G6 = FourierOperatorBlock2d(16 * self.width, 16 * self.width, 8, 8, 3, 3)
        self.G7 = FourierOperatorBlock2d(16 * self.width, 16 * self.width, 8, 8, 3, 3)
        self.G8 = FourierOperatorBlock2d(16 * self.width, 16 * self.width, 8, 8, 3, 3)
        self.G9 = FourierOperatorBlock2d(16 * self.width, 8 * self.width, 16, 16, 4, 4)
        self.G10 = FourierOperatorBlock2d(16 * self.width, 4 * self.width, 32, 32, 6, 6)
        self.G11 = FourierOperatorBlock2d(8 * self.width, 2 * self.width, 64, 64, 14, 14)
        self.G12 = FourierOperatorBlock2d(4 * self.width, self.width, 120, 120, 28, 28)
        self.fc1 = nn.Linear(1 * self.width, 2 * self.width)
        self.fc2 = nn.Linear(2 * self.width, 4)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)
        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        x_fc0 = F.pad(x_fc0, [0, self.padding, 0, self.padding])
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
        if self.padding != 0:
            x_c12 = x_c12[..., :-self.padding, :-self.padding]
        x_c12 = x_c12.permute(0, 2, 3, 1)
        x_fc1 = self.fc1(x_c12)
        x_fc1 = F.gelu(x_fc1)
        x_out = self.fc2(x_fc1)
        return x_out

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = (shape[0], shape[1], shape[2])
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)