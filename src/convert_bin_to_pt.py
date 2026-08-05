import numpy as np
import torch
import os
Outputpath_pt_trajectories = '/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/Trajectories/New/'
bins_file = '/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/Bin_TDHF/'
bin_files = sorted(os.listdir(bins_file))
for idx, bin_name in enumerate(bin_files, start=1):
    phis_list = []
    full_path = os.path.join(bins_file, bin_name)
    print('Reading:', full_path)
    with open(full_path, 'rb') as f:
        while True:
            try:
                nxyz = np.fromfile(f, dtype=np.intc, count=3)
                if nxyz.size < 3:
                    break
                dx = np.fromfile(f, dtype=np.double, count=1)
                if dx.size < 1:
                    break
                time = np.fromfile(f, dtype=np.double, count=1)
                if time.size < 1:
                    break
                nx, ny, nz = nxyz
                count = nx * nz * 6
                matrix = np.fromfile(f, dtype=np.float32, count=count)
                if matrix.size < count:
                    break
                matrix = matrix.reshape(nx * nz, 6)
                phist = np.zeros((nx, nz, 6), dtype=np.float32)
                for k in range(nx * nz):
                    ix = k // nz
                    iz = k % nz
                    phist[ix, iz] = matrix[k]
                phis_list.append(phist)
            except Exception as e:
                print('Stopped because of:', e)
                break
    phis = torch.from_numpy(np.array(phis_list, dtype=np.float32))
    print(f'data_shape = {phis.shape}')
    out_file = os.path.join(Outputpath_pt_trajectories, f'data_2d_{idx + 1335:04d}.pt')
    torch.save(phis, out_file)
    print('Saved:', out_file, '\n')