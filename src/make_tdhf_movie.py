import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import argparse
import os

def make_tdhf_movie_no_frames(trajectory_path: str, output_dir: str, trajectory_id: int, movie_name: str='trajectory.mp4', dt: float=9.0, arrow_step: int=2, arrow_scale: float=0.1):
    os.makedirs(output_dir, exist_ok=True)
    data = torch.load(trajectory_path, weights_only=False)
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    data = data.cpu().numpy()
    T, nx, ny, _ = data.shape
    rho = data[..., 0] + data[..., 1]
    jnx, jny = (data[..., 2], data[..., 4])
    jpx, jpy = (data[..., 3], data[..., 5])
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    Xs = X[::arrow_step, ::arrow_step]
    Ys = Y[::arrow_step, ::arrow_step]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmin, vmax = (rho.min(), rho.max())
    movie_path = os.path.join(output_dir, movie_name)
    writer = FFMpegWriter(fps=10, bitrate=1800)
    with writer.saving(fig, movie_path, dpi=150):
        for t in tqdm(range(T), desc='Rendering TDHF movie'):
            for ax in axes:
                ax.clear()
            axes[0].imshow(rho[t], cmap='plasma', vmin=vmin, vmax=vmax)
            axes[0].set_title('Total density $\\rho_n + \\rho_p$')
            axes[0].axis('off')
            axes[1].imshow(rho[t], cmap='Greys', alpha=0.25)
            axes[1].quiver(Xs, Ys, jnx[t][::arrow_step, ::arrow_step], -jny[t][::arrow_step, ::arrow_step], color='blue', angles='xy', scale_units='xy', scale=arrow_scale, width=0.003)
            axes[1].set_title('Neutron current $\\vec{j}_n$')
            axes[1].axis('off')
            axes[2].imshow(rho[t], cmap='Greys', alpha=0.25)
            axes[2].quiver(Xs, Ys, jpx[t][::arrow_step, ::arrow_step], -jpy[t][::arrow_step, ::arrow_step], color='red', angles='xy', scale_units='xy', scale=arrow_scale, width=0.003)
            axes[2].set_title('Proton current $\\vec{j}_p$')
            axes[2].axis('off')
            fig.suptitle(f'TDHF trajectory {trajectory_id} | t = {t * dt:.1f} fm/c', fontsize=18, fontweight='bold')
            writer.grab_frame()
    plt.close(fig)
    print(f' Movie created: {movie_path}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make TDHF movie from a single trajectory')
    parser.add_argument('trajectory_id', type=int, help='Trajectory number (e.g. 1, 2, 42, 1000)')
    parser.add_argument('--data_dir', type=str, default='/lustre/fsn1/projects/rech/lbf/umn29tg/TDHF_rot90', help='Directory containing TDHF .pt trajectories')
    parser.add_argument('--output_dir', type=str, default='/lustre/fsn1/projects/rech/lbf/umn29tg/Movies/tdhf_movie', help='Directory to save the movie')
    parser.add_argument('--dt', type=float, default=9.0, help='Time step in fm/c')
    args = parser.parse_args()
    trajectory_path = os.path.join(args.data_dir, f'data_2d_{args.trajectory_id:04d}.pt')
    movie_name = f'trajectory_{args.trajectory_id:04d}.mp4'
    print(f'▶ Making movie for trajectory {args.trajectory_id}')
    print(f'   Input:  {trajectory_path}')
    print(f'   Output: {movie_name}')
    make_tdhf_movie_no_frames(trajectory_path=trajectory_path, output_dir=args.output_dir, trajectory_id=args.trajectory_id, movie_name=movie_name, dt=args.dt)