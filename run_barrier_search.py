import torch
from inference_utils_Agress import load_inference_model, find_barrier
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Coulomb Barrier Search using UNO")
    parser.add_argument("--model_dir", type=str, default="/lustre/fsn1/projects/rech/lbf/umn29tg/TRAIN/UNO_MGPU_DL_Lightening/OutPuts/UNO_Agress_a100dev_PhysicsLoss__Epoc1000_LR1.0em4_Step3_Iter4_Nnull_Bz512_W20_GPU4_CPU9_3", help="Directory containing the model config")
    parser.add_argument("--ckpt", type=str, default="/lustre/fsn1/projects/rech/lbf/umn29tg/TRAIN/UNO_MGPU_DL_Lightening/OutPuts/UNO_Agress_a100dev_PhysicsLoss__Epoc1000_LR1.0em4_Step3_Iter4_Nnull_Bz512_W20_GPU4_CPU9_3/checkpoints/last.ckpt", help="Path to the model checkpoint")
    parser.add_argument("--data_dir", type=str, default="/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/TDHF90_NORMALIZED", help="Directory containing global_normalization_stats.pt")
    parser.add_argument("--traj", type=str, default="/lustre/fswork/projects/rech/lbf/umn29tg/ROOT/DATA/Trajectories/New/40Ca_90Zr_92.pt", help="Absolute path to the unnormalized .pt trajectory file")
    parser.add_argument("--ecm_min", type=float, default=40.0, help="Minimum Ecm for the search")
    parser.add_argument("--ecm_max", type=float, default=120.0, help="Maximum Ecm for the search")
    parser.add_argument("--n_step", type=int, default=60, help="Number of simulation steps to evolve")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model and normalization stats...")
    model, m, std = load_inference_model(args.model_dir, args.ckpt, args.data_dir, device)
    
    import os
    traj_path = args.traj
    if not os.path.exists(traj_path):
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        traj_path = os.path.join(repo_dir, "local_data", "40Ca_90Zr_92.pt")
    print(f"Loading initial density from {traj_path}...")
    full_traj = torch.load(traj_path, map_location='cpu')
    initial_density = full_traj[0:1]

    ecm_min = args.ecm_min
    ecm_max = args.ecm_max
    b = 0.0
    n_step = 100

    print("Starting automated Coulomb barrier search...")
    barrier, history = find_barrier(
        model=model,
        m=m,
        std=std,
        initial_density=initial_density,
        ecm_min=ecm_min,
        ecm_max=ecm_max,
        b=b,
        n_step=args.n_step,
        device=device,
        tolerance=0.5,
        max_iter=30,
        rho_cut=0.4
    )
    
    if barrier:
        print(f"\nFINAL BRACKETED BARRIER: {barrier:.3f} MeV")
    else:
        print("\nBarrier could not be found in the specified range.")
