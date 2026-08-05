import os
DATA_DIR = '/lustre/fsn1/projects/rech/lbf/umn29tg/TDHF_rot90'
N_TRAJ = 2766
missing = []
print(f'Auditing {N_TRAJ} augmented files in {DATA_DIR}...')
for i in range(1, N_TRAJ + 1):
    out_name = f'data_2d_{i + N_TRAJ:04d}.pt'
    if not os.path.exists(os.path.join(DATA_DIR, out_name)):
        missing.append(i)
if not missing:
    print(' SUCCESS: All 2766 augmented files exist on disk.')
else:
    print(f'️ FAILURE: Missing {len(missing)} files.')
    missing_str = ','.join(map(str, missing))
    print(f'Indices to re-run: {missing_str}')
    print(f'Re-run command: sbatch --array={missing_str} submit_augment.slurm')