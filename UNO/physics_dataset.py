import torch
import numpy as np
import scipy.io
import torch.nn as nn
import os
import operator
from functools import reduce
from functools import partial
from dataclasses import dataclass
import random
import gc
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class TrainConfig:
    numnodes: int
    numepochs: int
    gamma: float
    ratio: float
    stepsize: int
    use_GPU: bool
    wd: float
    learn: bool
    predstep: int
    iteration: int
    savemodel: bool
    save_partial_model: bool
    loadmodel: bool
    evolvetest: bool
    makemovie: bool
    batchsize: int
    TrajNum: int
    testnum: int
    width: int
    tag: str
    Checkpoint_name_part_save: str
    multi_GPU: bool
    initial_lr: float
    final_lr: float
    seed: int
    seed_number: int

class MatReader(object):

    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.file_path = file_path
        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]
        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
        if self.to_float:
            x = x.astype(np.float32)
        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.cuda()
        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

class LpLoss(object):

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class CustomLoss(object):

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(CustomLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y, numconstr):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        lambda0 = 1.0
        lambda1 = 1.0 / 100
        if self.reduction:
            if self.size_average:
                return lambda0 * torch.mean(diff_norms / y_norms) + lambda1 * numconstr
            else:
                return lambda0 * torch.sum(diff_norms / y_norms) + lambda1 * numconstr
        return lambda0 * diff_norms / y_norms + lambda1 * numconstr

    def __call__(self, x, y, numconstr):
        return self.rel(x, y, numconstr)

def iterative_process(model, xx, cfg):
    for _ in range(cfg.iteration):
        im_r = model(xx)
        xx = im_r
    return xx

def data_prepare(cfg: TrainConfig, inputpath):
    start_time = time.time()
    Train_dim = int(cfg.TrajNum * (1 - cfg.ratio))
    Val_dim = cfg.TrajNum - Train_dim
    random.seed(cfg.seed_number)
    rand_list = list(range(1, cfg.TrajNum + 1))
    random.shuffle(rand_list)
    Train_list = rand_list[:Train_dim]
    Val_list = rand_list[Train_dim:]
    a_train, u_train, a_val, u_val = ([], [], [], [])
    index = cfg.predstep * cfg.iteration
    print(f'→ Preparing data: {Train_dim} training trajectories, {Val_dim} validation trajectories')
    t0 = time.time()
    for ii in Train_list:
        phis = torch.load(f'{inputpath}/data_2d_{ii:04d}.pt', weights_only=False)
        a_train.append(phis[0:-index])
        u_train.append(phis[index:])
    print(f' Training data loaded in {time.time() - t0:.2f} s')
    t1 = time.time()
    for ii in Val_list:
        phis = torch.load(f'{inputpath}/data_2d_{ii:04d}.pt', weights_only=False)
        a_val.append(phis[0:-index])
        u_val.append(phis[index:])
    print(f' Validation data loaded in {time.time() - t1:.2f} s')
    t2 = time.time()
    a_train = np.concatenate(a_train, axis=0)
    u_train = np.concatenate(u_train, axis=0)
    a_val = np.concatenate(a_val, axis=0)
    u_val = np.concatenate(u_val, axis=0)
    print(f' Data concatenated in {time.time() - t2:.2f} s')
    t3 = time.time()
    indices = np.arange(len(a_train))
    np.random.seed(cfg.seed)
    np.random.shuffle(indices)
    a_train = a_train[indices]
    u_train = u_train[indices]
    indices_val = np.arange(len(a_val))
    np.random.seed(cfg.seed_number + 1)
    np.random.shuffle(indices_val)
    a_val = a_val[indices_val]
    u_val = u_val[indices_val]
    print(f' Data shuffled in {time.time() - t3:.2f} s')
    total_time = time.time() - start_time
    print(f' Total data preparation time: {total_time:.2f} seconds')
    return (a_train, u_train, a_val, u_val)

def normalizeData(numfields, a_train, u_train, a_val, u_val):
    m_a = np.mean(a_train, axis=(0, 1, 2))
    std_a = np.std(a_train, axis=(0, 1, 2))
    m_u = np.mean(u_train, axis=(0, 1, 2))
    std_u = np.std(u_train, axis=(0, 1, 2))
    for i in range(numfields):
        print(f'm_a[{i}]={m_a[i]:.9f}, std_a[{i}]={std_a[i]:.9f}')
        print(f'm_u[{i}]={m_u[i]:.9f}, std_u[{i}]={std_u[i]:.9f}')
    a_train -= m_a
    a_train /= std_a
    u_train -= m_u
    u_train /= std_u
    a_val -= m_a
    a_val /= std_a
    u_val -= m_u
    u_val /= std_u
    m = {'m_a': m_a, 'm_u': m_u}
    std = {'std_a': std_a, 'std_u': std_u}
    return (m, std, a_train, u_train, a_val, u_val)

def shuffleData(a_train, u_train, a_val, u_val):
    idx_train = np.arange(a_train.shape[0])
    np.random.shuffle(idx_train)
    a_train[:] = a_train[idx_train]
    u_train[:] = u_train[idx_train]
    idx_val = np.arange(a_val.shape[0])
    np.random.shuffle(idx_val)
    a_val[:] = a_val[idx_val]
    u_val[:] = u_val[idx_val]
    return (a_train, u_train, a_val, u_val)

def full_loop(rank, model, optimizer, scheduler, train_loader, valid_loader, outputpath, m, std, cfg: TrainConfig):
    train_losses = []
    val_losses = []
    LR = []
    ref = np.inf
    myloss = LpLoss(size_average=False)
    base_dir = os.path.join(outputpath, f'{cfg.Checkpoint_name_part_save}')
    os.makedirs(base_dir)
    for ep in range(cfg.numepochs):
        if cfg.multi_GPU == 1:
            torch.distributed.barrier()
            train_loader.sampler.set_epoch(ep)
        model.train()
        train_l2_sum = 0.0
        ntrain = 0
        for xx, yy in train_loader:
            optimizer.zero_grad()
            if cfg.use_GPU:
                device = rank if cfg.multi_GPU == 1 else 'cuda'
                xx = xx.to(device, non_blocking=True)
                yy = yy.to(device, non_blocking=True)
            batch_size = yy.size(0)
            ntrain += batch_size
            pred = iterative_process(model, xx, cfg)
            loss = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            train_l2_sum += loss.item()
            loss.backward()
            optimizer.step()
            del xx, yy, pred, loss
        gc.collect()
        train_loss_epoch = train_l2_sum / ntrain
        train_losses.append(train_loss_epoch)
        print(f'Epoch {ep}  Training Loss: {train_loss_epoch:.6f}', flush=True)
        if cfg.multi_GPU == 1:
            torch.distributed.barrier()
            valid_loader.sampler.set_epoch(ep)
        model.eval()
        val_l2_sum = 0.0
        nval = 0
        with torch.no_grad():
            for xx, yy in valid_loader:
                if cfg.use_GPU:
                    device = rank if cfg.multi_GPU == 1 else 'cuda'
                    xx = xx.to(device, non_blocking=True)
                    yy = yy.to(device, non_blocking=True)
                batch_size = yy.size(0)
                nval += batch_size
                pred = iterative_process(model, xx, cfg)
                loss = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
                val_l2_sum += loss.item()
                del xx, yy, pred, loss
        val_loss_epoch = val_l2_sum / nval
        val_losses.append(val_loss_epoch)
        current_lr = optimizer.param_groups[0]['lr']
        LR.append(current_lr)
        print(f'Epoch {ep}  Valid Loss: {val_loss_epoch:.6f}', flush=True)
        print(f'Learning rate: {current_lr}', flush=True)
        if val_loss_epoch < ref and ep != cfg.numepochs - 1:
            if cfg.multi_GPU == 0 and cfg.save_partial_model == 1:
                checkpoint = {'epoch': ep, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_loss': train_losses, 'val_loss': val_losses, 'Learning_Rate': LR, 'm': m, 'std': std}
                ref = val_loss_epoch
                torch.save(checkpoint, os.path.join(base_dir, 'model.checkpoint'))
                print(f'Checkpoint is saved in: {base_dir}')
        scheduler.step()
    return (train_losses, val_losses, LR)