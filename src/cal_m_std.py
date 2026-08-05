import torch
import os
import glob
import concurrent.futures
from tqdm import tqdm
num_channels = 6
data_dir = '/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/TDHF90'
output_file = '/lustre/fswork/projects/rech/lbf/umn29tg/UNO_TDHF_Project/output/dataset_stats.pt'
num_workers = min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

def process_single_file(filepath):
    try:
        if not os.path.exists(filepath):
            return None
        tensor = torch.load(filepath, map_location='cpu')
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        if tensor.shape[-1] != num_channels:
            return None
        tensor = tensor.double()
        tensor = tensor.view(-1, num_channels)
        local_sum = tensor.sum(dim=0)
        local_sum_sq = (tensor ** 2).sum(dim=0)
        local_pixels = tensor.shape[0]
        return (local_sum, local_sum_sq, local_pixels)
    except Exception:
        return None
if __name__ == '__main__':
    print(f'Scanning directory {data_dir}...')
    all_files = glob.glob(os.path.join(data_dir, 'data_2d_*.pt'))
    if not all_files:
        raise FileNotFoundError('No files found!')
    print(f'Detected {len(all_files)} files. Starting processing with {num_workers} parallel workers...')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    global_channel_sum = torch.zeros(num_channels, dtype=torch.float64)
    global_channel_sum_sq = torch.zeros(num_channels, dtype=torch.float64)
    global_total_pixels = 0
    valid_files = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results_generator = tqdm(executor.map(process_single_file, all_files, chunksize=20), total=len(all_files), desc='Calculating')
        for result in results_generator:
            if result is not None:
                local_sum, local_sum_sq, local_pixels = result
                global_channel_sum += local_sum
                global_channel_sum_sq += local_sum_sq
                global_total_pixels += local_pixels
                valid_files += 1
    if global_total_pixels == 0:
        raise RuntimeError('No valid data found to compute statistics!')
    mean = global_channel_sum / global_total_pixels
    variance = global_channel_sum_sq / global_total_pixels - mean ** 2
    variance = torch.clamp(variance, min=0.0)
    std = torch.sqrt(variance + 1e-12)
    mean = mean.float()
    std = std.float()
    print('\n--- Final Statistics ---')
    print(f'Processed files: {valid_files}/{len(all_files)}')
    print(f'Total pixels:    {global_total_pixels}')
    print(f'Mean:            {mean}')
    print(f'Std:             {std}')
    torch.save({'mean': mean, 'std': std}, output_file)
    print(f'\n Successfully saved statistics to {output_file}')