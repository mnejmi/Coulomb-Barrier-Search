import torch
import os
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
import yaml
DATA_DIR = '/lustre/fsn1/projects/rech/lbf/umn29tg/TDHF_rot90'
OUTPUT_DIR = '/lustre/fswork/projects/rech/lbf/umn29tg/UNO_TDHF_Project/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
NPROC = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))

def process_file(fpath):
    try:
        data = torch.load(fpath, map_location='cpu')
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        data = data.float()
        if data.ndim != 4 or data.shape[-1] != 6:
            print(f' Bad shape {data.shape} in {fpath}')
            return None
        x = data.reshape(-1, 6)
        return (x.sum(dim=0), (x ** 2).sum(dim=0), x.shape[0])
    except Exception as e:
        print(f'ERROR reading {fpath}: {e}')
        return None
if __name__ == '__main__':
    files = sorted(glob(os.path.join(DATA_DIR, 'data_2d_*.pt')))
    print('Files found:', len(files))
    print('CPU used:', NPROC)
    sum_c = None
    sum_sq_c = None
    total_count = 0
    ok_files = 0
    with Pool(NPROC) as pool:
        for result in tqdm(pool.imap_unordered(process_file, files), total=len(files)):
            if result is None:
                continue
            s, s2, n = result
            if sum_c is None:
                sum_c = s
                sum_sq_c = s2
            else:
                sum_c += s
                sum_sq_c += s2
            total_count += n
            ok_files += 1
    mean = sum_c / total_count
    std = torch.sqrt(sum_sq_c / total_count - mean ** 2)
    print('\n======================')
    print('Files processed:', ok_files)
    print('Mean:', mean)
    print('Std :', std)
    print('======================')
    torch.save({'mean': mean, 'std': std, 'count': int(total_count), 'files_used': ok_files}, os.path.join(OUTPUT_DIR, 'norm_stats.pt'))
    yaml_data = {'data_dir': DATA_DIR, 'files_total': len(files), 'files_used': ok_files, 'total_pixels': int(total_count), 'channels': 6, 'mean': mean.tolist(), 'std': std.tolist(), 'cpu_used': NPROC}
    with open(os.path.join(OUTPUT_DIR, 'norm_stats.yaml'), 'w') as f:
        yaml.dump(yaml_data, f)
    print('Saved stats + yaml.')