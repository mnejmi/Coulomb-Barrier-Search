import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def makeFrames(evolve_output, evolve_truth, Tevolve, outputpath, name, step, dataset, tagg, dt):
    result_rho = [evolve_truth[:, :, :, 0] + evolve_truth[:, :, :, 1], evolve_output[:, :, :, 0] + evolve_output[:, :, :, 1]]
    min_val_rho, max_val_rho = (np.amin(result_rho), np.amax(result_rho))
    for Time in range(Tevolve - 1):
        if dataset == '1D':
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(25, 12))
        elif dataset == 'TDHF':
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(25, 12), dpi=100)
        rho_output = evolve_output[Time, :, :, 0] + evolve_output[Time, :, :, 1]
        rho_truth = evolve_truth[Time, :, :, 0] + evolve_truth[Time, :, :, 1]
        im1 = axs[0].imshow(rho_output.squeeze(), vmin=min_val_rho, vmax=max_val_rho)
        im2 = axs[1].imshow(rho_truth.squeeze(), vmin=min_val_rho, vmax=max_val_rho)
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical', format='%.2f')
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical', format='%.2f')
        axs[0].axis('off')
        axs[1].axis('off')
        axs[0].set_title('Predicted Density')
        axs[1].set_title('Simulation Density')
        title = ['Time = ', str(dt * step * (Time + 1)), ' (fm/c)']
        title = ' '.join(title)
        if dataset == '1D':
            axs[1].text(40, 140, title, style='italic', fontsize=18, bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        elif dataset == 'TDHF':
            axs[1].text(14, 70, title, style='italic', fontsize=18, bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        filename = str(Time + 1)
        fig.savefig(f'{outputpath}/tag_{tagg}/n' + filename + '.png', bbox_inches='tight')
        plt.close(fig)
    os.system(f'ffmpeg -i {outputpath}/tag_{tagg}/n%d.png -c:v libx264 -r 30 -pix_fmt yuv420p {outputpath}/{name}.mp4')

def evolve_test(a_evolve, u_evolve, model, std, m, shape, step, dataset, use_GPU, multi_GPU, testnum):
    if dataset == '1D':
        dx = 0.8
    elif dataset == 'TDHF':
        dx = 0.9
        grid = 56
    mn = 939.565
    mp = 938.272
    Tevolve = {}
    for i in range(testnum):
        v_a = a_evolve[str(i)]
        v_u = u_evolve[str(i)]
        Tevolve[i] = v_a.shape[0]
        a_evolve[str(i)] = torch.tensor(v_a)
        u_evolve[str(i)] = torch.tensor(v_u)
    evolve = {}
    for i in range(testnum):
        evolve[str(i)] = torch.zeros([Tevolve[i], shape[1], shape[2], shape[3]], dtype=torch.float32)
    Treal = {}
    with torch.no_grad():
        for i in range(testnum):
            v_a = a_evolve[str(i)]
            predictions = v_a[0]
            v_evolve = evolve[str(i)]
            for time in range(Tevolve[i]):
                if step * time > Tevolve[i] - 1:
                    break
                if use_GPU == 1 and multi_GPU == 0:
                    v_evolve[time, :, :, :] = model(predictions[None, ...].cuda())
                if use_GPU == 0:
                    v_evolve[time, :, :, :] = model(predictions[None, ...].cpu())
                predictions = v_evolve[time, :, :, :].squeeze()
            Treal[i] = time
            evolve[str(i)] = v_evolve
    evolve_output = {}
    evolve_truth = {}
    errors = {}
    for i in range(testnum):
        evolve_output[str(i)] = np.empty([Treal[i], shape[1], shape[2], shape[3]])
        evolve_truth[str(i)] = np.empty([Treal[i], shape[1], shape[2], shape[3]])
        errors[str(i)] = torch.zeros([Treal[i], 5], dtype=torch.float32)
    for i in range(testnum):
        v_evolve_output = evolve_output[str(i)]
        v_evolve_truth = evolve_truth[str(i)]
        v_errors = errors[str(i)]
        v_evolve = evolve[str(i)]
        v_u_evolve = u_evolve[str(i)]
        for time in range(Treal[i]):
            for j in range(4):
                v_evolve_output[time, :, :, j] = v_evolve[time, :, :, j].cpu().squeeze().numpy() * std[j] + m[j]
                v_evolve_truth[time, :, :, j] = v_u_evolve[step * time, :, :, j].squeeze().numpy() * std[j] + m[j]
            time_real = step * (time + 1)
            pn_simu = dx ** 2 * (v_evolve_truth[time, :, :, 0] + v_evolve_truth[time, :, :, 1]).sum().item()
            pn_pred = dx ** 2 * (v_evolve_output[time, :, :, 0] + v_evolve_output[time, :, :, 1]).sum().item()
            nx, ny = (grid, grid)
            x = dx * np.linspace(-grid / 2, grid / 2 - 1, nx)
            y = dx * np.linspace(-grid / 2, grid / 2 - 1, ny)
            xv, yv = np.meshgrid(x, y)
            dx_c_simu = (dx ** 2 * xv * (mn * v_evolve_truth[time, :, :, 0] + mp * v_evolve_truth[time, :, :, 1])).sum().item() / (mn * v_evolve_truth[time, :, :, 0].sum().item() + mp * v_evolve_truth[time, :, :, 1].sum().item() * dx ** 2)
            dy_c_simu = (dx ** 2 * yv * (mn * v_evolve_truth[time, :, :, 0] + mp * v_evolve_truth[time, :, :, 1])).sum().item() / (mn * v_evolve_truth[time, :, :, 0].sum().item() + mp * v_evolve_truth[time, :, :, 1].sum().item() * dx ** 2)
            dx_c_pred = (dx ** 2 * xv * (mn * v_evolve_output[time, :, :, 0] + mp * v_evolve_output[time, :, :, 1])).sum().item() / (mn * v_evolve_truth[time, :, :, 0].sum().item() + mp * v_evolve_truth[time, :, :, 1].sum().item() * dx ** 2)
            dy_c_pred = (dx ** 2 * yv * (mn * v_evolve_output[time, :, :, 0] + mp * v_evolve_output[time, :, :, 1])).sum().item() / (mn * v_evolve_truth[time, :, :, 0].sum().item() + mp * v_evolve_truth[time, :, :, 1].sum().item() * dx ** 2)
            v_errors[time, 0] = time_real
            v_errors[time, 1] = ((dx_c_simu - dx_c_pred) ** 2 + (dy_c_simu - dy_c_pred) ** 2) ** 0.5
            v_errors[time, 3] = pn_simu
            v_errors[time, 4] = pn_pred
        errors[str(i)] = v_errors
    return (evolve_output, evolve_truth, Treal, errors)