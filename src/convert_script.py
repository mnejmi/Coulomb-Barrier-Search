import os
import glob
import torch
import h5py
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
SOURCE_DIR = '/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/TDHF90'
OUTPUT_FILE = '/lustre/fsn1/projects/rech/lbf/umn29tg/TDHF_combined90.h5'
N_WORKERS = min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
MAX_TRAJECTORIES = 5000
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)

def load_file(f):
    try:
        name = os.path.basename(f).replace('.pt', '')
        data = torch.load(f, map_location='cpu').numpy()
        return (name, data, None)
    except Exception as e:
        return (f, None, str(e))

def convert_to_hdf5():
    files = sorted(glob.glob(os.path.join(SOURCE_DIR, 'data_2d_*.pt')))
    if not files:
        raise FileNotFoundError(f'No .pt files found in {SOURCE_DIR}')
    files = files[:MAX_TRAJECTORIES]
    print(f'Found {len(files)} files (limited to {MAX_TRAJECTORIES})')
    print(f'Using {N_WORKERS} thread workers')
    with h5py.File(OUTPUT_FILE, 'w') as hf:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [executor.submit(load_file, f) for f in files]
            for i, future in enumerate(tqdm(as_completed(futures), total=len(files), desc='Converting')):
                name, data, error = future.result()
                if error is not None:
                    print(f'\nWarning: Could not load {name}. Error: {error}')
                    continue
                hf.create_dataset(name, data=data, compression='lzf')
                if i % 100 == 0:
                    hf.flush()
if __name__ == '__main__':
    convert_to_hdf5()