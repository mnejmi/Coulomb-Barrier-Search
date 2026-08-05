import torch
from pytorch_lightning.callbacks import Callback

class NetworkHealthMonitor(Callback):

    def __init__(self, dead_threshold=0.5, halt_on_explode=False):
        super().__init__()
        self.dead_threshold = dead_threshold
        self.halt_on_explode = halt_on_explode

    def on_train_epoch_end(self, trainer, pl_module):
        dead_layers = []
        exploded_layers = []
        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.numel() > 0:
                nans = torch.isnan(param).sum().item()
                infs = torch.isinf(param).sum().item()
                exploded = (torch.abs(param) > 10000.0).sum().item()
                if nans > 0 or infs > 0 or exploded > param.numel() * 0.01:
                    exploded_layers.append((name, nans, infs, exploded, param.numel()))
                dead_ratio = (torch.abs(param) < 1e-07).float().mean().item()
                if dead_ratio > self.dead_threshold:
                    dead_layers.append((name, dead_ratio))
        if exploded_layers:
            print('\n [CRITICAL WARNING] EXPLODED NEURONS DETECTED! ')
            for name, nans, infs, expl, n in exploded_layers:
                print(f'  - Layer {name}: {nans} NaNs, {infs} Infs, {expl} Exploded / {n} total')
            if self.halt_on_explode:
                print('Halting training due to exploded gradients.')
                trainer.should_stop = True
        if dead_layers:
            print(f'\n️ [WARNING] DEAD NEURONS DETECTED (> {self.dead_threshold * 100:.0f}% dead) ️')
            for name, ratio in dead_layers:
                print(f'  - Layer {name} is {ratio * 100:.1f}% dead!')