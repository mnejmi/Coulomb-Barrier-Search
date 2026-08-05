import torch
import math
import os
import numpy as np
from scipy.ndimage import label, center_of_mass, sum as ndi_sum
input_dir = '/lustre/fsn1/projects/rech/lbf/umn29tg/TDHF_rot90'
trajectory_number = 1336
tensor_ = torch.load(os.path.join(input_dir, f'data_2d_{trajectory_number:05d}.pt'))
hbc = 197.32164
h2ma = 20.7355298
e2 = 1.43989

def wood_saxon(x, y, z, N, Z, x0=0, y0=0, z0=0, rho_0=0.16):
    A = N + Z
    R = 1.08 * A ** (1 / 3)
    a = 0.5
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    return rho_0 / (1 + np.exp((r - R) / a))

def find_fragments_and_com(tensor_, rho_cut=1e-05, cell_area=0.9 * 0.9):
    if hasattr(tensor_, 'detach'):
        tensor_ = tensor_.detach().cpu().numpy()
    rho_p = tensor_[0, :, :, 0]
    rho_n = tensor_[0, :, :, 1]
    rho = rho_p + rho_n
    mask = rho >= rho_cut
    structure = np.ones((3, 3), dtype=int)
    labels, n_fragments = label(mask, structure=structure)
    results = []
    for lab in range(1, n_fragments + 1):
        x_com, z_com = center_of_mass(rho, labels, lab)
        Z = ndi_sum(rho_p, labels, lab) * cell_area
        N = ndi_sum(rho_n, labels, lab) * cell_area
        Z = int(np.round(Z))
        N = int(np.round(N))
        results.append({'label': lab, 'com': (x_com, z_com), 'Z': Z, 'N': N, 'A': Z + N})
    return (results, labels)

def fragments(rho, rho_cut=0.4, connectivity=8):
    mask = rho >= rho_cut
    if connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif connectivity == 8:
        structure = np.ones((3, 3), dtype=int)
    else:
        raise ValueError('connectivity must be 4 or 8')
    labels, n_fragments = label(mask, structure=structure)
    return (n_fragments, labels)

def evolve_test(a_initial, model, std, m, step, numfields, walltime, iteration, nx, nz, dx, dt):
    mn = 939.565
    mp = 938.272
    a_initial = torch.tensor(a_initial)
    Tevolve = int(walltime / (step * iteration * dt))
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

def predictions(tensor_, ecm, b):
    dx = 0.9
    nx = 56
    ny = 28
    nz = 56
    xm = (nx - 1) * dx / 2
    ym = (ny - 1) * dx / 2
    zm = (nz - 1) * dx / 2
    x = np.linspace(-xm, xm, nx)
    y = np.linspace(-ym, ym, ny)
    z = np.linspace(-zm, zm, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X2D, Z2D = np.meshgrid(x, z, indexing='ij')
    A1 = N1 + Z1
    x0_1, y0_1, z0_1 = (12, 0, 12)
    rho01n = rho0 * N1 / A1
    rho01p = rho0 * Z1 / A1
    N2 = 20
    Z2 = 20
    A2 = N2 + Z2
    x0_2, y0_2, z0_2 = (-12, 0, -12)
    rho02n = rho0 * N2 / A2
    rho02p = rho0 * Z2 / A2
    norm = np.sum(wood_saxon(X, Y, Z, N1, Z1, x0_1, y0_1, z0_1, rho01n)) * dx ** 3
    rho01n = rho01n * N1 / norm
    norm = np.sum(wood_saxon(X, Y, Z, N2, Z2, x0_2, y0_2, z0_2, rho02n)) * dx ** 3
    rho02n = rho02n * N2 / norm
    norm = np.sum(wood_saxon(X, Y, Z, N1, Z1, x0_1, y0_1, z0_1, rho01p)) * dx ** 3
    rho01p = rho01p * Z1 / norm
    norm = np.sum(wood_saxon(X, Y, Z, N2, Z2, x0_2, y0_2, z0_2, rho02p)) * dx ** 3
    rho02p = rho02p * Z2 / norm
    rho_n1 = wood_saxon(X, Y, Z, N1, Z1, x0_1, y0_1, z0_1, rho01n)
    rho_n2 = wood_saxon(X, Y, Z, N2, Z2, x0_2, y0_2, z0_2, rho02n)
    rho_p1 = wood_saxon(X, Y, Z, N1, Z1, x0_1, y0_1, z0_1, rho01p)
    rho_p2 = wood_saxon(X, Y, Z, N2, Z2, x0_2, y0_2, z0_2, rho02p)
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
    m = [0.04019840806722641, 0.0573577843606472, -6.10558799962746e-06, 5.421561581897549e-06, -2.3552975108032115e-06, 2.2208300833881367e-06]
    std = [0.13898997008800507, 0.1945505142211914, 0.010604926384985447, 0.014608636498451233, 0.01478016097098589, 0.020050009712576866]
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
hbc = 197.32164
h2ma = 20.7355298
nucleon_mass = hbc ** 2 / (2.0 * h2ma)
mu = nucleon_mass * A1 * A2 / (A1 + A2)
E_min = 3
E_max = 4

def search_E(x):
    return predictions(47.4, x) - 1.5
if __name__ == '__main__':
    rho_cut = 0.0001
    results, labels = find_fragments_and_com(tensor_, rho_cut=rho_cut)
    print(results)
    print(labels)

def initialisation_current(tensor_, ecm, rho_cut: float=0.0001):
    results, _ = find_fragments_and_com(tensor_, rho_cut=rho_cut)
    particle_1 = results[0]
    particle_2 = results[1]
    A1 = particle_1['A']
    A2 = particle_2['A']
    Z1 = particle_1['Z']
    Z2 = particle_2['Z']
    x0_1 = particle_1['com'][0]
    x0_2 = particle_2['com'][0]
    z0_1 = particle_1['com'][1]
    z0_2 = particle_2['com'][1]
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