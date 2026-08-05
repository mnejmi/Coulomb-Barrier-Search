import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utilities3 import *
from neural_operator import UNetNeuralOperator

def evolve_test(path_to_model, cfg: TrainConfig):
    numfields = 6
    dx = 0.9
    grid = 56
    mn = 939.565
    mp = 938.272
    model = UNetNeuralOperator(numfields + 2, cfg.width)
    checkpoint = torch.load('checkpoint.pt', map_location='cuda')
    phis_true = torch.load(f'sys_evolve_path', weights_only=False)
    phis_pred = torch.zeros_like(phis_true)
    Tevolve = phis_true.shape[0]
    with torch.no_grad():
        predictions = phis_true[0, ...]
        v_evolve = evolve[str(i)]
        for t in range(Tevolve):
            if cfg.stepsize * time > Tevolve - 1:
                break
            if cfg.use_GPU == 1 and cfg.multi_GPU == 0:
                phis_pred[t, :, :, :] = model(predictions[None, ...].cuda())
            if cfg.use_GPU == 0:
                phis_pred[t, :, :, :] = model(predictions[None, ...].cpu())
            predictions = phis_pred[t, :, :, :].squeeze()
    predictions_output = torch.zeros_like(predictions)
    errors = []
    for time in range(t):
        for j in range(4):
            predictions_output[time, :, :, j] = predictions[time, :, :, j].cpu().squeeze().numpy() * std[j] + m[j]
        time_real = cfg.stepsize * (time + 1)
        pn_simu = dx ** 2 * (phis_true[time_real, :, :, 0] + phis_true[time_real, :, :, 1]).sum().item()
        pn_pred = dx ** 2 * (predictions_output[time, :, :, 0] + predictions_output[time, :, :, 1]).sum().item()
        nx, ny = (grid, grid)
        x = dx * np.linspace(-grid / 2, grid / 2 - 1, nx)
        y = dx * np.linspace(-grid / 2, grid / 2 - 1, ny)
        xv, yv = np.meshgrid(x, y)
        dx_c_simu = (dx ** 2 * xv * (mn * phis_true[time_real, :, :, 0] + mp * phis_true[time_real, :, :, 1])).sum().item() / (mn * phis_true[time_real, :, :, 0].sum().item() + mp * phis_true[time_real, :, :, 1].sum().item() * dx ** 2)
        dy_c_simu = (dx ** 2 * yv * (mn * phis_true[time_real, :, :, 0] + mp * phis_true[time_real, :, :, 1])).sum().item() / (mn * phis_true[time_real, :, :, 0].sum().item() + mp * phis_true[time_real, :, :, 1].sum().item() * dx ** 2)
        dx_c_pred = (dx ** 2 * xv * (mn * predictions_output[time, :, :, 0] + mp * predictions_output[time, :, :, 1])).sum().item() / (mn * phis_true[time_real, :, :, 0].sum().item() + mp * phis_true[time_real, :, :, 1].sum().item() * dx ** 2)
        dy_c_pred = (dx ** 2 * yv * (mn * predictions_output[time, :, :, 0] + mp * predictions_output[time, :, :, 1])).sum().item() / (mn * phis_true[time_real, :, :, 0].sum().item() + mp * phis_true[time_real, :, :, 1].sum().item() * dx ** 2)
        v_errors[time, 0] = time_real
        v_errors[time, 1] = ((dx_c_simu - dx_c_pred) ** 2 + (dy_c_simu - dy_c_pred) ** 2) ** 0.5
        v_errors[time, 3] = pn_simu
        v_errors[time, 4] = pn_pred
        errors += [v_errors]
    return (predictions_output, v_errors, errors)

def makeFrames(evolve_output, evolve_truth, Tevolve, outputpath, name, step, tagg, dt):
    result_rho = [evolve_truth[:, :, :, 0] + evolve_truth[:, :, :, 1], evolve_output[:, :, :, 0] + evolve_output[:, :, :, 1]]
    min_val_rho, max_val_rho = (np.amin(result_rho), np.amax(result_rho))
    for Time in range(Tevolve - 1):
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
        axs[1].text(14, 70, title, style='italic', fontsize=18, bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        filename = str(Time + 1)
        fig.savefig(f'{outputpath}/tag_{tagg}/n' + filename + '.png', bbox_inches='tight')
        plt.close(fig)
    os.system(f'ffmpeg -i {outputpath}/tag_{tagg}/n%d.png -c:v libx264 -r 30 -pix_fmt yuv420p {outputpath}/{name}.mp4')