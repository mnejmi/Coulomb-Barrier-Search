import torch
import math
import numpy as np
from scipy.ndimage import label, center_of_mass
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
from utilities3 import *

def fragments(rho, rho_cut=0.4):
    if isinstance(rho, torch.Tensor):
        rho = rho.detach().cpu().numpy()
    mask = rho >= rho_cut
    labeled_array, num_fragments = label(mask)
    return num_fragments

def detect_fragments(rho, threshold=0.01):
    mask = rho > threshold
    labeled, num = label(mask)
    return (labeled, num)

def iterative_process(model, xx, cfg):
    current_state = xx
    for _ in range(cfg.iteration):
        pred = model(current_state)
        if cfg.predict_residual:
            current_state = current_state + pred
        else:
            current_state = pred
    return current_state

def COMs(data: torch.Tensor, threshold: float=0.01):
    rho = data[0, :, :, 0] + data[0, :, :, 1]
    rho_np = rho.detach().cpu().numpy()
    mask = rho_np > threshold
    labeled, num_fragments = label(mask)
    if num_fragments == 0:
        raise ValueError('No fragments found. Try lowering threshold.')
    coms = center_of_mass(rho_np, labeled, range(1, num_fragments + 1))
    coms = sorted(coms, key=lambda x: x[0])
    result = {}
    for i, (x, z) in enumerate(coms):
        result[f'x0_{i + 1}'] = float(x)
        result[f'z0_{i + 1}'] = float(z)
    return result

def prepare_current(cfg, data: torch.Tensor, ecm: float=0.0, b: float=0.0):
    dx = 0.9
    nx, ny = (56, 56)
    rho_p = data[0, :, :, 0]
    rho_n = data[0, :, :, 1]
    mid = nx // 2
    P1 = int(round((rho_p[mid:, :mid].sum() * dx * dx).item()))
    N1 = int(round((rho_n[mid:, :mid].sum() * dx * dx).item()))
    P2 = int(round((rho_p[:mid, mid:].sum() * dx * dx).item()))
    N2 = int(round((rho_n[:mid, mid:].sum() * dx * dx).item()))
    A1 = P1 + N1
    A2 = P2 + N2
    print(f'P1={P1}, N1={N1}, P2={P2}, N2={N2}, A1={A1}, A2={A2}')
    result = COMs(data=data, threshold=0.01)
    if len(result) < 2:
        raise ValueError('Less than 2 fragments detected')
    x1, z1 = (result['x0_1'], result['z0_1'])
    x2, z2 = (result['x0_2'], result['z0_2'])
    dx1 = (x1 - x2) * dx
    dz1 = (z1 - z2) * dx
    roft = math.sqrt(dx1 ** 2 + dz1 ** 2)
    if roft < 1e-06:
        raise ValueError('Fragments overlap → roft ~ 0')
    dix = dx1 / roft
    diz = dz1 / roft
    e2 = 1.43989
    hbc = 197.32164
    h2ma = 20.7355298
    nucleon_mass = hbc ** 2 / (2.0 * h2ma)
    xmu = nucleon_mass * A1 * A2 / (A1 + A2)
    vrel = math.sqrt(2.0 * ecm / xmu) if ecm > 0 else 0.0
    xli = xmu * vrel * b / hbc
    Z1, Z2 = (P1, P2)
    ec = e2 * Z1 * Z2 / roft
    if ec >= ecm:
        raise ValueError(f'Insufficient energy: Coulomb energy ({ec:.2f} MeV) exceeds E_cm ({ecm:.2f} MeV).')
    vrel_d = math.sqrt(2.0 * (ecm - ec) / xmu)
    v1 = A2 / (A1 + A2) * vrel_d
    v2 = A1 / (A1 + A2) * vrel_d
    b_d = xli * hbc / (xmu * vrel_d)
    sint = b_d / roft
    if sint > 1.0:
        raise ValueError('Impact parameter exceeds distance between fragments (sint > 1). Unphysical geometry.')
    cost = math.sqrt(1.0 - sint ** 2)
    vx1 = -v1 * (dix * cost - diz * sint)
    vz1 = +v1 * (dix * sint + diz * cost)
    vx2 = +v2 * (dix * cost - diz * sint)
    vz2 = -v2 * (dix * sint + diz * cost)
    rho_p1 = torch.zeros((56, 56), dtype=data.dtype, device=data.device)
    rho_n1 = torch.zeros((56, 56), dtype=data.dtype, device=data.device)
    rho_p2 = torch.zeros((56, 56), dtype=data.dtype, device=data.device)
    rho_n2 = torch.zeros((56, 56), dtype=data.dtype, device=data.device)
    rho_p2[:28, 28:] = data[0, :28, 28:, 0]
    rho_n2[:28, 28:] = data[0, :28, 28:, 1]
    rho_p1[28:, :28] = data[0, 28:, :28, 0]
    rho_n1[28:, :28] = data[0, 28:, :28, 1]
    factor = hbc / (2 * h2ma)
    curnx1 = vx1 * rho_n1 * factor
    curnx2 = vx2 * rho_n2 * factor
    curpx1 = vx1 * rho_p1 * factor
    curpx2 = vx2 * rho_p2 * factor
    curnz1 = vz1 * rho_n1 * factor
    curnz2 = vz2 * rho_n2 * factor
    curpz1 = vz1 * rho_p1 * factor
    curpz2 = vz2 * rho_p2 * factor
    curnx = curnx1 + curnx2
    curpx = curpx1 + curpx2
    curnz = curnz1 + curnz2
    curpz = curpz1 + curpz2
    data[0, :, :, 2] = curpx
    data[0, :, :, 3] = curnx
    data[0, :, :, 4] = curpz
    data[0, :, :, 5] = curnz
    return (data, roft)

def get_final_state(data, roft):
    ...

def evolve_test(m, std, model, cfg: TrainConfig, i=3):
    numfields = 6
    dx = 0.9
    grid = 56
    mn = 939.565
    mp = 938.272
    phis_true = torch.load(f'{inputpath_tensor}/data_2d_{i:05d}.pt', weights_only=False)
    if isinstance(phis_true, np.ndarray):
        phis_true = torch.tensor(phis_true, dtype=torch.float32)
    else:
        phis_true = phis_true.float()
    m = torch.tensor(m, dtype=torch.float32).reshape(1, 1, 6)
    std = torch.tensor(std, dtype=torch.float32).reshape(1, 1, 6)
    if cfg.use_gpu:
        phis_true = phis_true.cuda()
        m = m.cuda()
        std = std.cuda()
        model = model.cuda().float()
    else:
        model = model.float()
    T = phis_true.shape[0]
    phis_pred = torch.zeros_like(phis_true)
    with torch.no_grad():
        predictions = (phis_true[0] - m) / std
        phis_pred[0] = predictions[None, ...][0]
        for t in range(1, T):
            if cfg.stepsize * t >= T:
                print(f't = {t}')
                break
            out = model(predictions[None, ...])
            phis_pred[t] = out[0]
            predictions = out[0]
    predictions_output = phis_pred[:t + 1] * std + m
    v_errors = torch.zeros((t + 1, 5), dtype=torch.float32, device=phis_true.device)
    errors = []
    x = dx * np.linspace(-grid / 2, grid / 2 - 1, grid)
    y = dx * np.linspace(-grid / 2, grid / 2 - 1, grid)
    xv, yv = np.meshgrid(x, y)
    xv = torch.tensor(xv, dtype=torch.float32, device=phis_true.device)
    yv = torch.tensor(yv, dtype=torch.float32, device=phis_true.device)
    for time in range(t + 1):
        time_real = cfg.stepsize * time
        if time_real >= T:
            break
        rho_simu = phis_true[time_real, ..., 0] + phis_true[time_real, ..., 1]
        rho_pred = predictions_output[time, ..., 0] + predictions_output[time, ..., 1]
        pn_simu = dx ** 2 * rho_simu.sum().item()
        pn_pred = dx ** 2 * rho_pred.sum().item()
        dx_c_simu = (dx ** 2 * xv * rho_simu).sum().item() / (rho_simu.sum().item() * dx ** 2)
        dy_c_simu = (dx ** 2 * yv * rho_simu).sum().item() / (rho_simu.sum().item() * dx ** 2)
        dx_c_pred = (dx ** 2 * xv * rho_pred).sum().item() / (rho_pred.sum().item() * dx ** 2)
        dy_c_pred = (dx ** 2 * yv * rho_pred).sum().item() / (rho_pred.sum().item() * dx ** 2)
        v_errors[time, 0] = time_real
        v_errors[time, 1] = math.sqrt((dx_c_simu - dx_c_pred) ** 2 + (dy_c_simu - dy_c_pred) ** 2)
        v_errors[time, 3] = pn_simu
        v_errors[time, 4] = pn_pred
        errors.append(v_errors[time].clone())
    return (predictions_output, v_errors, errors, rho_simu, rho_pred)

def predictions_output(m, std, model, cfg: 'TrainConfig', a_initial=None, i=3, n_step: int=60, inputpath_tensor: str=None, rho_cut: float=0.4, Normalise=False):
    numfields = 6
    dx = 0.9
    grid = 56
    mn = 939.565
    mp = 938.272
    if a_initial is None:
        phis_true = torch.load(f'{inputpath_tensor}/data_2d_{i:05d}.pt', weights_only=False)
    else:
        phis_true = a_initial
    if isinstance(phis_true, np.ndarray):
        phis_true = torch.tensor(phis_true, dtype=torch.float32)
    else:
        phis_true = phis_true.float()
    if not isinstance(m, torch.Tensor):
        m = torch.tensor(m, dtype=torch.float32)
    else:
        m = m.detach().clone().float()
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=torch.float32)
    else:
        std = std.detach().clone().float()
    m = m.reshape(1, 1, 6)
    std = std.reshape(1, 1, 6)
    if cfg.use_gpu:
        phis_true = phis_true.cuda()
        m = m.cuda()
        std = std.cuda()
        model = model.cuda().float()
    else:
        model = model.float()
    target_p_sum = torch.sum(phis_true[0, :, :, 0])
    target_n_sum = torch.sum(phis_true[0, :, :, 1])
    if a_initial is None:
        T = phis_true.shape[0]
        _, H, W, C = phis_true.shape
        phis_pred = torch.zeros((T, H, W, C), dtype=phis_true.dtype, device=phis_true.device)
    else:
        _, H, W, C = phis_true.shape
        phis_pred = torch.zeros((n_step, H, W, C), dtype=phis_true.dtype, device=phis_true.device)
    compt = 1
    inert_t = 1
    if a_initial is None:
        with torch.no_grad():
            predictions = (phis_true[0] - m) / std
            phis_pred[0] = predictions
            for t in range(1, T):
                if cfg.stepsize * t >= T:
                    print(f't = {t}')
                    break
                diff = model(predictions[None, ...])[0]
                out_1 = predictions + diff
                out_phys = out_1 * std + m
                if Normalise:
                    current_p_sum = torch.sum(out_phys[:, :, 0])
                    out_phys[:, :, 0] *= target_p_sum / (current_p_sum + 1e-10)
                    current_n_sum = torch.sum(out_phys[:, :, 1])
                    out_phys[:, :, 1] *= target_n_sum / (current_n_sum + 1e-10)
                out = (out_phys - m) / std
                phis_pred[t] = out
                predictions = out
    elif cfg.predict_residual:
        with torch.no_grad():
            predictions = (phis_true[0] - m) / std
            phis_pred[0] = predictions
            for t in range(1, n_step):
                diff = model(predictions[None, ...])[0]
                out_1 = predictions + diff
                if Normalise:
                    out_phys = out_1 * std + m
                    current_p_sum = torch.sum(out_phys[:, :, 0])
                    out_phys[:, :, 0] *= target_p_sum / (current_p_sum + 1e-10)
                    current_n_sum = torch.sum(out_phys[:, :, 1])
                    out_phys[:, :, 1] *= target_n_sum / (current_n_sum + 1e-10)
                    out_1 = (out_phys - m) / std
                phis_pred[t] = out_1
                predictions = out_1
    else:
        with torch.no_grad():
            predictions = (phis_true[0] - m) / std
            phis_pred[0] = predictions
            for t in range(1, n_step):
                current_state = predictions
                for _ in range(cfg.iteration):
                    current_state = model(current_state[None, ...])[0]
                if Normalise:
                    out_phys = current_state * std + m
                    current_p_sum = torch.sum(out_phys[:, :, 0])
                    out_phys[:, :, 0] *= target_p_sum / (current_p_sum + 1e-10)
                    current_n_sum = torch.sum(out_phys[:, :, 1])
                    out_phys[:, :, 1] *= target_n_sum / (current_n_sum + 1e-10)
                    current_state = (out_phys - m) / std
                phis_pred[t] = current_state
                predictions = current_state
    try:
        predictions_output = phis_pred[:t + 1] * std + m
    except UnboundLocalError:
        predictions_output = phis_pred * std + m
    num = fragments(predictions_output[-1, :, :, 0] + predictions_output[-1, :, :, 1], rho_cut=rho_cut)
    return (predictions_output, num)

def search_E(x, make_movie=False):
    data, _ = prepare_current(cfg=cfg, data=phis_true, ecm=x, b=0)
    predictions_output, num = predictions_output(m, std, model, a_initial=data, cfg=cfg, i=3, n_step=60)
    if make_movie:
        ...
    return num

def compute_barrier(make_movie, phis_pred, ecmm, ecmM, b):
    fussion = False
    tolerance = 0.01
    ecm_min = ecmm
    ecm_max = ecmM
    ecm = ecm_max
    while not fussion:
        ...
    num = fragments(phis_pred[-1, :, :, 0] + phis_pred[-1, :, :, 1], rho_cut=0.4)
    return barrier_positions

def bisection_method(func, a, b, tolerance=1e-05, max_iterations=40):
    if func(a) * func(b) >= 0:
        raise ValueError('Function must have opposite signs at the endpoints of the interval')
    iteration = 0
    while (b - a) / 2 > tolerance and iteration < max_iterations:
        midpoint = (a + b) / 2
        if func(midpoint) == 0:
            return midpoint
        elif func(a) * func(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        iteration += 1
    return (a + b) / 2

def search_E(x):
    return predictions(47.4, x) - 1.5

def evolve_test(a_initial, model, step, walltime, iteration, nx, nz, dx, dt, m, std, cfg, numfields=6):
    mn = 939.565
    mp = 938.272
    a_initial = torch.tensor(a_initial)
    Tevolve = int(walltime / (cfg.step * cfg.iteration * dt))
    evolve = torch.zeros([Tevolve, nx, nz, numfields], dtype=torch.float32)
    Treal = {}
    with torch.no_grad():
        num_simu = (a_initial[:, :, 0] + a_initial[:, :, 1]).sum().item() * dx ** 2
        predictions = torch.empty([nx, nz, numfields])
        for i in range(6):
            predictions[:, :, i] = (a_initial[:, :, i] - m[i]) / std[i]
        for time in range(Tevolve):
            evolve[time, :, :, :] = model(predictions[None, ...].cuda())
            sl_pred0 = evolve[time, :, :, 0].cpu().squeeze().numpy() * std[0] + m[0]
            sl_pred1 = evolve[time, :, :, 1].cpu().squeeze().numpy() * std[1] + m[1]
            sl_pred0[sl_pred0 < 0] = 0
            sl_pred1[sl_pred1 < 0] = 0
            num_simu = (a_initial[:, :, 0] + a_initial[:, :, 1]).sum().item() * dx ** 2
            num_pred = (sl_pred0 + sl_pred1).sum().item() * dx ** 2
            evolve_real0 = torch.empty(nx, nz)
            evolve_real1 = torch.empty(nx, nz)
            evolve_real0 = evolve[time, :, :, 0].cpu().squeeze() * std[0] + m[0]
            evolve_real1 = evolve[time, :, :, 1].cpu().squeeze() * std[1] + m[1]
            evolve_real0 = evolve_real0 * num_simu / num_pred
            evolve_real1 = evolve_real1 * num_simu / num_pred
            evolve[time, :, :, 0] = (evolve_real0 - m[0]) / std[0]
            evolve[time, :, :, 1] = (evolve_real1 - m[1]) / std[1]
            predictions = evolve[time, :, :, :].squeeze()
    evolve_output = np.empty([Tevolve + 1, nx, nz, numfields])
    errors = torch.zeros([Tevolve + 1, 7], dtype=torch.float32)
    evolve_output[0, :, :, :] = a_initial[:, :, :].numpy()
    for time in range(Tevolve):
        for j in range(numfields):
            evolve_output[time + 1, :, :, j] = evolve[time, :, :, j].cpu().squeeze().numpy() * std[j] + m[j]
        time_real = step * (time + 1)
        pn_pred = dx ** 2 * (evolve_output[time, :, :, 0] + evolve_output[time, :, :, 1]).sum().item()
        x = dx * np.linspace(-nx / 2, nx / 2 - 1, nx)
        y = -dx * np.linspace(-nz / 2, nz / 2 - 1, nz)
        xv, yv = np.meshgrid(x, y)
        dx_c_pred = (dx ** 2 * xv * (mn * evolve_output[time, :, :, 0] + mp * evolve_output[time, :, :, 1])).sum().item() / (mn * a_initial[:, :, 0].sum().item() + mp * a_initial[:, :, 1].sum().item() * dx ** 2)
        dy_c_pred = (dx ** 2 * yv * (mn * evolve_output[time, :, :, 0] + mp * evolve_output[time, :, :, 1])).sum().item() / (mn * a_initial[:, :, 0].sum().item() + mp * a_initial[:, :, 1].sum().item() * dx ** 2)
        l_pred = (-(xv - dx_c_pred) * (evolve_output[time, :, :, 2] + evolve_output[time, :, :, 3]) - (yv - dy_c_pred) * (evolve_output[time, :, :, 4] + evolve_output[time, :, :, 5])).sum().item() * dx ** 2
        errors[time, 0] = time_real
        errors[time, 1] = pn_pred
        errors[time, 2] = l_pred
        density = evolve_output[time, :, :, 0].squeeze() + evolve_output[time, :, :, 1].squeeze()
        den_thres = 0.1
        if np.any(density[0, :] > den_thres):
            evolve_output = evolve_output[:time, :, :, :]
            break
        if np.any(density[-1, :] > den_thres):
            evolve_output = evolve_output[:time, :, :, :]
            break
        if np.any(density[:, 0] > den_thres):
            evolve_output = evolve_output[:time, :, :, :]
            break
        if np.any(density[:, -1] > den_thres):
            evolve_output = evolve_output[:time, :, :, :]
            break
    density = evolve_output[-1, :, :, 0].squeeze() + evolve_output[-1, :, :, 1].squeeze()
    frag_num = fragments(density)
    return (evolve_output, time - 1, errors, frag_num)

def predictions(RHO, ecm, b=0):
    dx = 0.9
    nx = 56
    ny = 28
    nz = 56
    xm = (nx - 1) * dx / 2
    ym = (ny - 1) * dx / 2
    zm = (nz - 1) * dx / 2
    rho_n = np.sum(rho_n1 + rho_n2, axis=1) * dx
    rho_p = np.sum(rho_p1 + rho_p2, axis=1) * dx
    e2 = 1.43989
    hbc = 197.32164
    h2ma = 20.7355298
    nucleon_mass = hbc ** 2 / (2.0 * h2ma)
    xmu = nucleon_mass * A1 * A2 / (A1 + A2)
    vrel = math.sqrt(2.0 * ecm / xmu)
    xli = xmu * vrel * b / hbc
    dix = x0_1 - x0_2
    diz = z0_1 - z0_2
    roft = math.sqrt(dix ** 2 + diz ** 2)
    dix = dix / roft
    diz = diz / roft
    ec = e2 * Z1 * Z2 / roft
    if ec > ecm:
        print('Not enough energy to reach this distance')
        exit(1)
    vrel_d = math.sqrt(2.0 * (ecm - ec) / xmu)
    v1 = A2 / (A1 + A2) * vrel_d
    v2 = A1 / (A1 + A2) * vrel_d
    b_d = xli * hbc / (xmu * vrel_d)
    sint = b_d / roft
    cost = math.sqrt(1.0 - sint ** 2)
    vx1 = -v1 * (dix * cost - diz * sint)
    vz1 = -v1 * (dix * sint + diz * cost)
    vx2 = +v2 * (dix * cost - diz * sint)
    vz2 = +v2 * (dix * sint + diz * cost)
    curnx1 = vx1 * rho_n1 * (hbc / (2 * h2ma))
    curnx2 = vx2 * rho_n2 * (hbc / (2 * h2ma))
    curpx1 = vx1 * rho_p1 * (hbc / (2 * h2ma))
    curpx2 = vx2 * rho_p2 * (hbc / (2 * h2ma))
    curnz1 = vz1 * rho_n1 * (hbc / (2 * h2ma))
    curnz2 = vz2 * rho_n2 * (hbc / (2 * h2ma))
    curpz1 = vz1 * rho_p1 * (hbc / (2 * h2ma))
    curpz2 = vz2 * rho_p2 * (hbc / (2 * h2ma))
    curnx = np.sum(curnx1 + curnx2, axis=1) * dx
    curpx = np.sum(curpx1 + curpx2, axis=1) * dx
    curnz = np.sum(curnz1 + curnz2, axis=1) * dx
    curpz = np.sum(curpz1 + curpz2, axis=1) * dx
    arrays = [rho_p, rho_n, curpx, curnx, curpz, curnz]
    a_initial = np.stack(arrays, axis=2)
    evolve_output, Tevolve, errors, frag_num = evolve_test(a_initial, model, std, m, predstep, numfields, walltime, iteration, nx, nz, dx, dt)
    return frag_num

def bisection_method(func, a, b, tolerance=1e-05, max_iterations=40):
    if func(a) * func(b) >= 0:
        raise ValueError('Function must have opposite signs at the endpoints of the interval')
    iteration = 0
    while (b - a) / 2 > tolerance and iteration < max_iterations:
        midpoint = (a + b) / 2
        if func(midpoint) == 0:
            return midpoint
        elif func(a) * func(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        iteration += 1
    return (a + b) / 2
numfields = 6
width = 64
learn = 0.0005
wd = 0.0005
stepsize = 70
gamma = 0.2
numfields = 6
predstep = 5
iteration = 2
dt = 9
walltime = 1500
device = torch.device('cuda')

def search_E(x):
    return predictions(47.4, x) - 1.5

def make_movie(evolve_output: torch.Tensor, output_dir: str, trajectory_id: int, trajectory_path: str=None, compare_with_tdhf: bool=True, show_currents: bool=True, movie_name: str=None, dt: float=9.0, stepsize: int=1, arrow_step: int=2, arrow_scale: float=0.1, Trjnum=11000):
    os.makedirs(output_dir, exist_ok=True)
    evo_uno = evolve_output.detach().cpu().numpy()
    if evo_uno.ndim == 3:
        evo_uno = evo_uno[None, ...]
    T_pred, nx, ny, _ = evo_uno.shape
    rho_uno = evo_uno[..., 0] + evo_uno[..., 1]
    if show_currents:
        jn_x_u, jn_y_u = (evo_uno[..., 2], evo_uno[..., 4])
        jp_x_u, jp_y_u = (evo_uno[..., 3], evo_uno[..., 5])
    if compare_with_tdhf and trajectory_path is not None:
        path = os.path.join(trajectory_path, f'data_2d_{trajectory_id:05d}.pt')
        evo_true = torch.load(path, map_location='cpu')
        if not isinstance(evo_true, torch.Tensor):
            evo_true = torch.tensor(evo_true)
        evo_true = evo_true.numpy()
        T_true = evo_true.shape[0]
        rho_true = evo_true[..., 0] + evo_true[..., 1]
        if show_currents:
            jn_x_t, jn_y_t = (evo_true[..., 2], evo_true[..., 4])
            jp_x_t, jp_y_t = (evo_true[..., 3], evo_true[..., 5])
        T = min(T_pred, T_true // stepsize)
        vmin = min(rho_true.min(), rho_uno.min())
        vmax = max(rho_true.max(), rho_uno.max())
    else:
        compare_with_tdhf = False
        T = T_pred
        vmin, vmax = (rho_uno.min(), rho_uno.max())
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    Xs = X[::arrow_step, ::arrow_step]
    Ys = Y[::arrow_step, ::arrow_step]
    nrows = 2 if compare_with_tdhf else 1
    ncols = 3 if show_currents else 1
    base_height = 4.5 if nrows == 1 else 3.5
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, base_height * nrows))
    axes = np.atleast_2d(axes)
    if movie_name is None:
        movie_name = f'trajectory_{Trjnum}_{trajectory_id}.mp4'
    if not movie_name.endswith('.mp4'):
        movie_name += '.mp4'
    movie_path = os.path.join(output_dir, movie_name)
    writer = FFMpegWriter(fps=10, bitrate=1800, extra_args=['-pix_fmt', 'yuv420p'])
    with writer.saving(fig, movie_path, dpi=150):
        for t in tqdm(range(T), desc='Rendering movie'):
            for ax in axes.flat:
                ax.clear()
            if compare_with_tdhf:
                true_index = t * stepsize
                axes[0, 0].imshow(rho_true[true_index], cmap='plasma', vmin=vmin, vmax=vmax)
                axes[0, 0].set_title('TDHF Total Density', pad=15)
                axes[0, 0].axis('off')
                if show_currents:
                    axes[0, 1].imshow(rho_true[true_index], cmap='Greys', alpha=0.25)
                    axes[0, 1].quiver(Xs, Ys, jn_x_t[true_index][::arrow_step, ::arrow_step], -jn_y_t[true_index][::arrow_step, ::arrow_step], angles='xy', scale_units='xy', color='blue', scale=arrow_scale)
                    axes[0, 1].set_title('TDHF Neutron Current', pad=15)
                    axes[0, 1].axis('off')
                    axes[0, 2].imshow(rho_true[true_index], cmap='Greys', alpha=0.25)
                    axes[0, 2].quiver(Xs, Ys, jp_x_t[true_index][::arrow_step, ::arrow_step], -jp_y_t[true_index][::arrow_step, ::arrow_step], angles='xy', scale_units='xy', color='red', scale=arrow_scale)
                    axes[0, 2].set_title('TDHF Proton Current', pad=15)
                    axes[0, 2].axis('off')
            row = 1 if compare_with_tdhf else 0
            axes[row, 0].imshow(rho_uno[t], cmap='plasma', vmin=vmin, vmax=vmax)
            axes[row, 0].set_title('UNO Total Density', pad=15)
            axes[row, 0].axis('off')
            if show_currents:
                axes[row, 1].imshow(rho_uno[t], cmap='Greys', alpha=0.25)
                axes[row, 1].quiver(Xs, Ys, jn_x_u[t][::arrow_step, ::arrow_step], -jn_y_u[t][::arrow_step, ::arrow_step], angles='xy', scale_units='xy', color='blue', scale=arrow_scale)
                axes[row, 1].set_title('UNO Neutron Current', pad=15)
                axes[row, 1].axis('off')
                axes[row, 2].imshow(rho_uno[t], cmap='Greys', alpha=0.25)
                axes[row, 2].quiver(Xs, Ys, jp_x_u[t][::arrow_step, ::arrow_step], -jp_y_u[t][::arrow_step, ::arrow_step], angles='xy', scale_units='xy', color='red', scale=arrow_scale)
                axes[row, 2].set_title('UNO Proton Current', pad=15)
                axes[row, 2].axis('off')
            time_phys = dt * t
            if compare_with_tdhf:
                title = f'TDHF vs UNO | Time = {time_phys:.1f} fm/c'
            else:
                title = f'UNO Prediction | Time = {time_phys:.1f} fm/c'
            fig.suptitle(title, fontsize=18, fontweight='bold', y=0.95)
            plt.subplots_adjust(top=0.75, bottom=0.05, hspace=0.35, wspace=0.2)
            writer.grab_frame()
    plt.close(fig)
    print(f' Movie created successfully: {movie_path}')

def make_movie2(evolve_output: torch.Tensor, output_dir: str, trajectory_id: int, trajectory_path: str=None, compare_with_tdhf: bool=True, show_currents: bool=True, movie_name: str=None, dt: float=9.0, stepsize: int=1, arrow_step: int=2, arrow_scale: float=0.1, Trjnum=11000):
    os.makedirs(output_dir, exist_ok=True)
    evo_uno = evolve_output.detach().cpu().numpy()
    if evo_uno.ndim == 3:
        evo_uno = evo_uno[None, ...]
    T_pred, nx, ny, _ = evo_uno.shape
    rho_uno = evo_uno[..., 0] + evo_uno[..., 1]
    if show_currents:
        jn_x_u, jn_y_u = (evo_uno[..., 2], evo_uno[..., 4])
        jp_x_u, jp_y_u = (evo_uno[..., 3], evo_uno[..., 5])
    if compare_with_tdhf and trajectory_path is not None:
        path = os.path.join(trajectory_path, f'data_2d_{trajectory_id:05d}.pt')
        evo_true = torch.load(path, map_location='cpu')
        if not isinstance(evo_true, torch.Tensor):
            evo_true = torch.tensor(evo_true)
        evo_true = evo_true.numpy()
        rho_true = evo_true[..., 0] + evo_true[..., 1]
        if show_currents:
            jn_x_t, jn_y_t = (evo_true[..., 2], evo_true[..., 4])
            jp_x_t, jp_y_t = (evo_true[..., 3], evo_true[..., 5])
        T = min(T_pred, evo_true.shape[0] // stepsize)
        vmin = min(rho_true.min(), rho_uno.min())
        vmax = max(rho_true.max(), rho_uno.max())
    else:
        compare_with_tdhf = False
        T = T_pred
        vmin, vmax = (rho_uno.min(), rho_uno.max())
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    Xs = X[::arrow_step, ::arrow_step]
    Ys = Y[::arrow_step, ::arrow_step]
    nrows = 2 if compare_with_tdhf else 1
    ncols = 3 if show_currents else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 6))
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(top=0.85, bottom=0.05, hspace=0.3, wspace=0.2)
    if movie_name is None:
        movie_name = f'trajectory_{Trjnum}_{trajectory_id}.mp4'
    if not movie_name.endswith('.mp4'):
        movie_name += '.mp4'
    movie_path = os.path.join(output_dir, movie_name)
    writer = FFMpegWriter(fps=10, bitrate=1800, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
    with writer.saving(fig, movie_path, dpi=150):
        for t in tqdm(range(T), desc='Rendering movie'):
            for ax in axes.flat:
                ax.clear()
            if compare_with_tdhf:
                idx = t * stepsize
                axes[0, 0].imshow(rho_true[idx], cmap='plasma', vmin=vmin, vmax=vmax)
                axes[0, 0].set_title('TDHF Density')
                axes[0, 0].axis('off')
                if show_currents:
                    axes[0, 1].imshow(rho_true[idx], cmap='Greys', alpha=0.25)
                    axes[0, 1].quiver(Xs, Ys, jn_x_t[idx][::arrow_step, ::arrow_step], -jn_y_t[idx][::arrow_step, ::arrow_step], angles='xy', scale_units='xy', color='blue', scale=arrow_scale)
                    axes[0, 1].set_title('TDHF Neutron Current', pad=15)
                    axes[0, 1].axis('off')
                    axes[0, 2].imshow(rho_true[idx], cmap='Greys', alpha=0.25)
                    axes[0, 2].quiver(Xs, Ys, jp_x_t[idx][::arrow_step, ::arrow_step], -jp_y_t[idx][::arrow_step, ::arrow_step], angles='xy', scale_units='xy', color='red', scale=arrow_scale)
                    axes[0, 2].set_title('TDHF Proton Current', pad=15)
                    axes[0, 2].axis('off')
            row = 1 if compare_with_tdhf else 0
            axes[row, 0].imshow(rho_uno[t], cmap='plasma', vmin=vmin, vmax=vmax)
            axes[row, 0].set_title('UNO Density')
            axes[row, 0].axis('off')
            if show_currents:
                axes[row, 1].imshow(rho_uno[t], cmap='Greys', alpha=0.25)
                axes[row, 1].quiver(Xs, Ys, jn_x_u[t][::arrow_step, ::arrow_step], -jn_y_u[t][::arrow_step, ::arrow_step], angles='xy', scale_units='xy', color='blue', scale=arrow_scale)
                axes[row, 1].set_title('UNO Neutron Current', pad=15)
                axes[row, 1].axis('off')
                axes[row, 2].imshow(rho_uno[t], cmap='Greys', alpha=0.25)
                axes[row, 2].quiver(Xs, Ys, jp_x_u[t][::arrow_step, ::arrow_step], -jp_y_u[t][::arrow_step, ::arrow_step], angles='xy', scale_units='xy', color='red', scale=arrow_scale)
                axes[row, 2].set_title('UNO Proton Current', pad=15)
                axes[row, 2].axis('off')
            title = f'TDHF vs UNO | t = {dt * t:.1f} fm/c' if compare_with_tdhf else f'UNO Prediction | t = {dt * t:.1f} fm/c'
            fig.suptitle(title, fontsize=16, fontweight='bold')
            writer.grab_frame()
    plt.close(fig)
    print(f' Movie created: {movie_path}')