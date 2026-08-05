import os
import sys
import glob
import torch
import numpy as np
import math
from scipy.ndimage import label, center_of_mass
import scipy.ndimage as ndimage


def setup_paths(model_dir, use_uno=True):
    """Add the model directory and its UNO or UNO_multi_GPU module to sys.path so Lightning can unpickle the model properly."""
    multi_gpu_path = os.path.join(model_dir, 'UNO_multi_GPU')
    uno_path = os.path.join(model_dir, 'UNO')
    
    # 1. Insert at the VERY FRONT of sys.path to shadow any other model folders
    if model_dir in sys.path:
        sys.path.remove(model_dir)
    sys.path.insert(0, model_dir)
    
    # Clean up both first
    if multi_gpu_path in sys.path:
        sys.path.remove(multi_gpu_path)
    if uno_path in sys.path:
        sys.path.remove(uno_path)
        
    # Add only the targeted path to prevent namespace conflicts (e.g. architecture.py shading)
    if use_uno:
        sys.path.insert(0, uno_path)
    else:
        sys.path.insert(0, multi_gpu_path)
    
    # 2. Aggressively clear any previously cached versions of the module from other cells!
    to_delete = [m for m in sys.modules.keys() if m.startswith('UNO_multi_GPU') or m.startswith('UNO') or m == 'architecture']
    for m in to_delete:
        del sys.modules[m]

def load_inference_model(model_dir, checkpoint_path, data_dir, device):
    """Loads the config, normalization stats, and PyTorch Lightning Checkpoint."""
    
    # Inject ConfigDict into __main__ so PyTorch can unpickle it properly
    class ConfigDict(dict):
        def __getattr__(self, key):
            if key in self:
                val = self[key]
                if isinstance(val, dict):
                    return ConfigDict(val)
                return val
            raise AttributeError(f"No such attribute: {key}")
        def __setattr__(self, key, value):
            self[key] = value
    sys.modules['__main__'].ConfigDict = ConfigDict

    # Load config file (YAML) with fallbacks
    import yaml, glob
    config_file = os.path.join(model_dir, 'config_uno.yaml')
    if not os.path.exists(config_file):
        yaml_files = glob.glob(os.path.join(model_dir, "*.yaml"))
        if yaml_files:
            config_file = yaml_files[0]
        else:
            config_file = os.path.join(data_dir, 'config_uno.yaml')
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        cfg = ConfigDict(config_dict)
        print(f"  [Config] Loaded from: {config_file}")
    else:
        cfg = ConfigDict({'width': 64, 'data_dir': data_dir, 'model_type': 'phys', 'dropout': 0.0})
        print(f"  [Config] No YAML found, using defaults (width=64)")

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'hyper_parameters' in ckpt and isinstance(ckpt['hyper_parameters'], dict):
        hp = ckpt['hyper_parameters']
        # Don't let checkpoint's data_dir override local paths!
        local_data_dir = cfg.get('data_dir', data_dir)
        for k, v in hp.items():
            cfg[k] = v
        cfg['data_dir'] = local_data_dir  # Restore local data_dir
        print(f"  [Config] Loaded {len(hp)} hyper_parameters from checkpoint")
            
    # Auto-detect width from state_dict to prevent size mismatch
    state_dict = ckpt.get('state_dict', {})
    if 'model.fc0.weight' in state_dict:
        detected_width = state_dict['model.fc0.weight'].shape[0]
        cfg['width'] = detected_width
        print(f"  [Config] Auto-detected width = {detected_width} from checkpoint fc0.weight")
    
    # Detect V6 vs V7 from checkpoint keys
    has_time_proj = any('time_proj_16' in k for k in state_dict.keys())
    has_ln3 = any('ln3' in k for k in state_dict.keys())
    ckpt_is_v7 = has_time_proj and has_ln3
    print(f"  [Checkpoint] Architecture detection: time_proj_16={has_time_proj}, ln3={has_ln3} → {'V7' if ckpt_is_v7 else 'V6'}")
    
    # V6 training code (train_transformer_v6_pinn.py) does NOT implement predict_residual.
    # The config value is a leftover from copy-paste. Force False for V6.
    if not ckpt_is_v7 and cfg.get('predict_residual', False):
        print(f"  [Config] OVERRIDE: predict_residual=True → False (V6 training code does not implement residual prediction)")
        cfg['predict_residual'] = False
    
    # Print key summary
    model_keys = [k for k in state_dict.keys() if k.startswith('model.')]
    print(f"  [Checkpoint] Total state_dict keys: {len(state_dict)}, model.* keys: {len(model_keys)}")

    stats_path = os.path.join(data_dir, 'global_normalization_stats.pt')
    if not os.path.exists(stats_path):
        stats_path = os.path.join(data_dir, 'stats.pt')
    stats = torch.load(stats_path, map_location='cpu')
    m, std = stats['m'], stats['std']
    print(f"  [Stats] m = {m.tolist()}")
    print(f"  [Stats] std = {std.tolist()}")

    model = None
    model_class_name = None
    
    if ckpt_is_v7:
        # Try V7 first since checkpoint has V7 keys
        try:
            from train_transformer_v7_ultimate import UNOT_V7_Lightning
            model = UNOT_V7_Lightning.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False)
            model_class_name = "UNOT_V7_Lightning"
        except Exception as e:
            print(f"  [WARN] V7 loading failed: {e}")
    else:
        # Try V6 first since checkpoint has V6 keys
        try:
            from train_transformer_v6_pinn import UNOT_V6_Lightning
            model = UNOT_V6_Lightning.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False)
            model_class_name = "UNOT_V6_Lightning"
        except Exception as e:
            print(f"  [WARN] V6 loading failed: {e}")
    
    # Fallback: try the other architecture
    if model is None:
        try:
            from train_transformer_v7_ultimate import UNOT_V7_Lightning
            model = UNOT_V7_Lightning.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False)
            model_class_name = "UNOT_V7_Lightning (fallback)"
        except Exception as e:
            try:
                from train_transformer_v6_pinn import UNOT_V6_Lightning
                model = UNOT_V6_Lightning.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False)
                model_class_name = "UNOT_V6_Lightning (fallback)"
            except Exception as e2:
                print(f"  [ERROR] Failed loading V7: {e}")
                print(f"  [ERROR] Failed loading V6: {e2}")

    if model is None:
        try:
            setup_paths(model_dir, use_uno=True)
            sys.path.insert(0, os.path.join(BASE_DIR, 'UNO'))
            from UNO.train import LightningNeuralOperator
            model = LightningNeuralOperator.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False)
            model_class_name = "LightningNeuralOperator"
        except Exception:
            setup_paths(model_dir, use_uno=False)
            sys.path.insert(0, os.path.join(BASE_DIR, 'UNO_multi_GPU'))
            from UNO_multi_GPU.train import UNOT_Lightning
            model = UNOT_Lightning.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False)
            model_class_name = "UNOT_Lightning"

    print(f"  [Model] Loaded as: {model_class_name}")
    print(f"  [Model] cfg.width={getattr(cfg, 'width', '?')}, cfg.pred_step={getattr(cfg, 'pred_step', '?')}, cfg.predict_residual={getattr(cfg, 'predict_residual', False)}")
    print(f"  [Model] Inner model class: {type(model.model).__name__}")
    
    # Verify loaded weights match checkpoint
    loaded_sd = model.state_dict()
    missing = [k for k in state_dict if k not in loaded_sd]
    unexpected = [k for k in loaded_sd if k not in state_dict]
    if missing:
        print(f"  [WARN] {len(missing)} checkpoint keys NOT loaded: {missing[:5]}...")
    if unexpected:
        print(f"  [WARN] {len(unexpected)} model keys NOT in checkpoint (randomly initialized!): {unexpected[:5]}...")
    
    model = model.to(device)
    model.eval()
    return model, m, std

def run_autoregressive_inference(model, m, std, traj_path, device, is_normalized=False):
    """Loads a trajectory and runs full inference over all time steps."""
    phis = torch.load(traj_path, map_location='cpu') # [T, H, W, 6]
    T_actual = phis.shape[0]
    
    m_dev = m.to(device)
    std_dev = std.to(device)
    
    if is_normalized:
        # File is already normalized. We need to un-normalize it to physical units for the ground truths.
        phis_phys = phis.to(device) * std_dev + m_dev
        a_norm = phis[0].unsqueeze(0).to(torch.float32).to(device)
        a_phys = phis_phys[0].unsqueeze(0)
    else:
        # File is raw physical data
        phis_phys = phis.to(device)
        a_phys = phis[0].unsqueeze(0).to(torch.float32).to(device)
        a_norm = (a_phys - m_dev) / std_dev
    
    predictions = [a_phys.cpu().squeeze(0)]
    ground_truths = [a_phys.cpu().squeeze(0)]
    
    curr_x = a_norm.clone()
    curr_phys = a_phys.clone() # Keep track of previous physical frame for residuals
    
    # Get the exact true initial particle count to conserve separately
    num_protons_true = a_phys[0, :, :, 0].sum().item()
    num_neutrons_true = a_phys[0, :, :, 1].sum().item()
    
    # Check model configuration
    pred_step = getattr(model.cfg, 'pred_step', 3) 
    predict_res = getattr(model.cfg, 'predict_residual', False)
    
    with torch.no_grad():
        for target_idx in range(pred_step, T_actual, pred_step):
            pred_N = model.model(curr_x)
            
            # Un-normalize (check if predicting residual)
            if predict_res:
                # Training does: pred_t = current_x + raw_pred (in normalized space)
                pred_norm = curr_x + pred_N
                pred_N_phys = pred_norm * std_dev + m_dev
            else:
                pred_N_phys = pred_N * std_dev + m_dev
                
            # --- PHYSICS ENFORCEMENT: Renormalization ---
            # 1. Vacuum cutoff (remove negative noise artifacts)
            pred_N_phys[:, :, :, 0] = torch.clamp(pred_N_phys[:, :, :, 0], min=0.0)
            pred_N_phys[:, :, :, 1] = torch.clamp(pred_N_phys[:, :, :, 1], min=0.0)
            
            # 2. Scale back to exact original mass (protons and neutrons independently)
            num_protons_pred = pred_N_phys[:, :, :, 0].sum().item()
            num_neutrons_pred = pred_N_phys[:, :, :, 1].sum().item()
            
            if num_protons_pred > 0:
                scale_factor_p = num_protons_true / num_protons_pred
                pred_N_phys[:, :, :, 0] *= scale_factor_p
                
            if num_neutrons_pred > 0:
                scale_factor_n = num_neutrons_true / num_neutrons_pred
                pred_N_phys[:, :, :, 1] *= scale_factor_n
                
            # 3. Update the state for the next autoregressive step
            curr_phys = pred_N_phys.clone()
            curr_x = (curr_phys - m_dev) / std_dev
                
            true_N_phys = phis_phys[target_idx].unsqueeze(0) # already physical
            
            predictions.append(pred_N_phys.cpu().squeeze(0))
            ground_truths.append(true_N_phys.cpu().squeeze(0))
            
    predictions = torch.stack(predictions, dim=0)
    ground_truths = torch.stack(ground_truths, dim=0)
    
    return predictions, ground_truths, T_actual

def count_fragments(rho, rho_cut=0.4):
    if isinstance(rho, torch.Tensor):
        rho = rho.detach().cpu().numpy()
        
    # Smooth the density specifically for peak-finding to eliminate noisy flat plateaus
    # (We still use the true 'rho' to calculate the actual physical densities later!)
    smoothed_rho = ndimage.gaussian_filter(rho, sigma=1.5)
    
    # Find local maxima on the smoothed surface
    local_max = (ndimage.maximum_filter(smoothed_rho, size=5) == smoothed_rho)
    
    # Remove background noise peaks using the original unsmoothed threshold
    background = (rho < rho_cut)
    valid_peaks = local_max ^ (local_max & background)
    
    labeled_peaks, num_peaks = label(valid_peaks)
    max_rho = np.max(rho)
    
    if num_peaks <= 1:
        return num_peaks, max_rho, 0.0
        
    # Multiple peaks detected. Sort them to find the two main nuclei
    peak_coords = np.argwhere(valid_peaks)
    peak_values = [rho[p[0], p[1]] for p in peak_coords]
    
    sorted_indices = np.argsort(peak_values)[::-1]
    p1 = peak_coords[sorted_indices[0]]
    p2 = peak_coords[sorted_indices[1]]
    
    # Extract the density along the straight line connecting the two peaks (the neck)
    num_points = max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
    if num_points == 0:
        return 1, max_rho, 0.0
        
    x_indices = np.linspace(p1[0], p2[0], num_points, dtype=int)
    y_indices = np.linspace(p1[1], p2[1], num_points, dtype=int)
    
    line_densities = rho[x_indices, y_indices]
    neck_density = np.min(line_densities)
    
    min_peak = min(rho[p1[0], p1[1]], rho[p2[0], p2[1]])
    
    # If the density at the saddle point drops below 50% of the peaks, they are separated.
    if neck_density < 0.5 * min_peak:
        return 2, max_rho, neck_density
    else:
        # The saddle point is thick, they are a single deformed/fused blob
        return 1, max_rho, neck_density

def compute_COMs(data, threshold=None):
    if data.dim() == 3:
        data = data.unsqueeze(0)
    
    rho = data[0, :, :, 0] + data[0, :, :, 1]
    rho_np = rho.detach().cpu().numpy()
    
    # Use a dynamic relative threshold (10% of the peak density) to absolutely guarantee 
    # we ignore any background noise floor left over from unnormalization!
    dynamic_threshold = np.max(rho_np) * 0.1
    mask = rho_np > dynamic_threshold
    labeled, num_fragments = label(mask)
    
    if num_fragments < 2:
        raise ValueError(f"Expected 2 fragments in initial state, found {num_fragments}.")
    
    coms = center_of_mass(rho_np, labeled, range(1, num_fragments + 1))
    coms = sorted(coms, key=lambda x: x[0])
    
    result = {}
    for i, (x, z) in enumerate(coms):
        result[f"x0_{i+1}"] = float(x)
        result[f"z0_{i+1}"] = float(z)
    return result

def prepare_current(data, ecm, b=0.0, dx=0.9):
    data = data.clone()
    if data.dim() == 3:
        data = data.unsqueeze(0)
    
    nx, ny = 56, 56
    mid = nx // 2
    
    rho_p = data[0, :, :, 0]
    rho_n = data[0, :, :, 1]
    
    # Calculate masses in all 4 quadrants
    quads = [
        (slice(mid, None), slice(None, mid)),   # bottom-left
        (slice(None, mid), slice(mid, None)),   # top-right
        (slice(None, mid), slice(None, mid)),   # top-left
        (slice(mid, None), slice(mid, None)),   # bottom-right
    ]
    
    quad_masses = []
    for sx, sz in quads:
        p = int(round((rho_p[sx, sz].sum() * dx * dx).item()))
        n = int(round((rho_n[sx, sz].sum() * dx * dx).item()))
        quad_masses.append((sx, sz, p, n))
        
    # Sort quadrants by total mass A = p + n (descending)
    quad_masses.sort(key=lambda x: x[2] + x[3], reverse=True)
    
    # The two nuclei must be the top 2
    nuc1 = quad_masses[0]
    nuc2 = quad_masses[1]
    
    P1, N1 = nuc1[2], nuc1[3]
    P2, N2 = nuc2[2], nuc2[3]
    
    A1, A2 = P1 + N1, P2 + N2
    Z1, Z2 = P1, P2
    
    result = compute_COMs(data)
    cx1, cz1 = result["x0_1"], result["z0_1"]
    cx2, cz2 = result["x0_2"], result["z0_2"]
    
    # Helper to check if a coordinate falls in a slice
    def in_slice(val, s):
        start = 0 if s.start is None else s.start
        stop = 56 if s.stop is None else s.stop
        return start <= val < stop
        
    # Match the sorted COMs to the mass-sorted quadrants
    if in_slice(cx1, nuc1[0]) and in_slice(cz1, nuc1[1]):
        x1, z1 = cx1, cz1
        x2, z2 = cx2, cz2
    else:
        x1, z1 = cx2, cz2
        x2, z2 = cx1, cz1
    
    dx1 = (x1 - x2) * dx
    # Dynamically assign A1 and A2 based on the fragments found
    # We must ensure we assign the correct mass to the correct nucleus, otherwise the center-of-mass will drift.
    dx = 0.9; dy = 0.9
    # Better to sum the unnormalized density mask to find the true Z
    rho_np = (data[0, :, :, 0] + data[0, :, :, 1]).cpu().numpy()
    mask = rho_np > 0.05
    l_mask, n_labels = label(mask)
    mask1 = (l_mask == 1)
    Z1_sum = data[0, :, :, 0].cpu().numpy()[mask1].sum() * dx * dy
    if Z1_sum > 30: # 90Zr has Z=40
        A1 = 90
        A2 = 40
    else:           # 40Ca has Z=20
        A1 = 40
        A2 = 90

    # Reduced mass (using MeV)
    h2ma = 20.7355298
    hbc = 197.32164
    nucleon_mass = hbc ** 2 / (2.0 * h2ma)
    xmu = nucleon_mass * A1 * A2 / (A1 + A2)

    # Center of mass distance
    cx1, cz1 = x1, z1
    cx2, cz2 = x2, z2
    dx1 = (cx1 - cx2) * dx
    dz1 = (cz1 - cz2) * dx
    roft = math.sqrt(dx1**2 + dz1**2)
    dix = dx1 / roft
    diz = dz1 / roft

    # Initial velocities and scaling
    e2 = 1.43989
    Z_1 = 40 if A1 == 90 else 20
    Z_2 = 20 if A1 == 90 else 40
    ec = e2 * Z_1 * Z_2 / roft
    
    if ec >= ecm:
        raise ValueError(f"Coulomb energy ({ec:.2f} MeV) ≥ E_cm ({ecm:.2f} MeV).")
    
    # Calculate velocities
    vrel_d = math.sqrt(2.0 * max(0.0, ecm - ec) / xmu)
    v1 = A2 / (A1 + A2) * vrel_d
    v2 = A1 / (A1 + A2) * vrel_d
    
    xli = xmu * vrel_d * b / hbc
    b_d = xli * hbc / (xmu * vrel_d)
    sint = b_d / roft
    cost = math.sqrt(max(0.0, 1.0 - sint**2))
    
    vx1 = -v1 * (dix * cost - diz * sint)
    vz1 = -v1 * (dix * sint + diz * cost)
    vx2 = +v2 * (dix * cost - diz * sint)
    vz2 = +v2 * (dix * sint + diz * cost)
    
    device = data.device
    dtype = data.dtype
    
    rho_p1 = torch.zeros((56, 56), dtype=dtype, device=device)
    rho_n1 = torch.zeros((56, 56), dtype=dtype, device=device)
    rho_p2 = torch.zeros((56, 56), dtype=dtype, device=device)
    rho_n2 = torch.zeros((56, 56), dtype=dtype, device=device)
    
    sx1, sz1 = nuc1[0], nuc1[1]
    rho_p1[sx1, sz1] = data[0, sx1, sz1, 0]
    rho_n1[sx1, sz1] = data[0, sx1, sz1, 1]
    
    sx2, sz2 = nuc2[0], nuc2[1]
    rho_p2[sx2, sz2] = data[0, sx2, sz2, 0]
    rho_n2[sx2, sz2] = data[0, sx2, sz2, 1]
    
    factor = hbc / (2 * h2ma)
    
    data[0, :, :, 2] = (vx1 * rho_p1 + vx2 * rho_p2) * factor
    data[0, :, :, 3] = (vx1 * rho_n1 + vx2 * rho_n2) * factor
    data[0, :, :, 4] = (vz1 * rho_p1 + vz2 * rho_p2) * factor
    data[0, :, :, 5] = (vz1 * rho_n1 + vz2 * rho_n2) * factor
    
    return data, roft

def run_inference_from_state(model, m, std, initial_state_phys, device, n_step=100, verbose=False):
    m_dev = m.to(device)
    std_dev = std.to(device)
    
    if initial_state_phys.dim() == 3:
        initial_state_phys = initial_state_phys.unsqueeze(0)
    initial_state_phys = initial_state_phys.to(dtype=torch.float32, device=device)
    
    a_norm = (initial_state_phys - m_dev) / std_dev
    
    if verbose:
        rho_init = initial_state_phys[0, :, :, 0] + initial_state_phys[0, :, :, 1]
        print(f"    [INF] Initial state PHYSICAL: rho_max={rho_init.max().item():.4f}, rho_sum={rho_init.sum().item():.1f}")
        for ch in range(6):
            ch_norm = a_norm[0, :, :, ch]
            ch_phys = initial_state_phys[0, :, :, ch]
            print(f"    [INF]   ch{ch} phys: [{ch_phys.min().item():.4f}, {ch_phys.max().item():.4f}]  "
                  f"norm: [{ch_norm.min().item():.2f}, {ch_norm.max().item():.2f}]  "
                  f"absmax_norm={ch_norm.abs().max().item():.2f}")
    
    predictions = [initial_state_phys.cpu().squeeze(0)]
    
    curr_x = a_norm.clone()
    curr_phys = initial_state_phys.clone() # Keep track of previous physical frame for residuals
    
    # Get initial mass to conserve
    num_protons_true = initial_state_phys[0, :, :, 0].sum().item()
    num_neutrons_true = initial_state_phys[0, :, :, 1].sum().item()
    
    predict_res = getattr(model.cfg, 'predict_residual', False)
    
    with torch.no_grad():
        for target_idx in range(1, n_step):
            pred_N = model.model(curr_x)
            
            if verbose and target_idx <= 3:
                print(f"    [INF] Step {target_idx}: pred_N min={pred_N.min().item():.6f} max={pred_N.max().item():.6f} "
                      f"abs_mean={pred_N.abs().mean().item():.6f}")
            
            if predict_res:
                # Training does: pred_t = current_x + raw_pred (in normalized space)
                # So we must add residual in normalized space, THEN unnormalize
                pred_norm = curr_x + pred_N
                pred_N_phys = pred_norm * std_dev + m_dev
            else:
                pred_N_phys = pred_N * std_dev + m_dev
                
            # --- PHYSICS ENFORCEMENT ---
            pred_N_phys[:, :, :, 0] = torch.clamp(pred_N_phys[:, :, :, 0], min=0.0)
            pred_N_phys[:, :, :, 1] = torch.clamp(pred_N_phys[:, :, :, 1], min=0.0)
            
            num_protons_pred = pred_N_phys[:, :, :, 0].sum().item()
            num_neutrons_pred = pred_N_phys[:, :, :, 1].sum().item()
            
            if num_protons_pred > 0:
                scale_factor_p = num_protons_true / num_protons_pred
                pred_N_phys[:, :, :, 0] *= scale_factor_p
                
            if num_neutrons_pred > 0:
                scale_factor_n = num_neutrons_true / num_neutrons_pred
                pred_N_phys[:, :, :, 1] *= scale_factor_n
                
            if verbose and target_idx <= 3:
                rho_step = pred_N_phys[0, :, :, 0] + pred_N_phys[0, :, :, 1]
                print(f"    [INF] Step {target_idx}: rho_max={rho_step.max().item():.4f} "
                      f"rho_sum={rho_step.sum().item():.1f}")
                
            # Update state
            curr_phys = pred_N_phys.clone()
            curr_x = (curr_phys - m_dev) / std_dev
            
            predictions.append(pred_N_phys.cpu().squeeze(0))
            
    if verbose:
        rho_final = predictions[-1][:, :, 0] + predictions[-1][:, :, 1]
        print(f"    [INF] Final frame: rho_max={rho_final.max().item():.4f}")
            
    return torch.stack(predictions, dim=0)

def evaluate_ecm(model, m, std, initial_density, ecm, b, n_step, device, rho_cut=0.4):
    # Unnormalize initial_density (which comes normalized from the dataset)
    # into physical units so prepare_current calculates momentum correctly.
    # run_inference_from_state will correctly re-normalize the combined physical state!
    initial_density_phys = (initial_density.cpu() * std.cpu()) + m.cpu()
    
    data_with_currents, _ = prepare_current(initial_density_phys, ecm, b)
    trajectory = run_inference_from_state(model, m, std, data_with_currents, device, n_step, verbose=True)
    
    # Extract total density (Protons + Neutrons) of the final frame
    rho_final = trajectory[-1, :, :, 0] + trajectory[-1, :, :, 1]
    
    if isinstance(rho_final, torch.Tensor):
        rho_final_np = rho_final.detach().cpu().numpy()
    else:
        rho_final_np = rho_final
        
    # Check if density touches the simulation walls (margin of 2 pixels)
    margin = 2
    touches_wall = (
        (rho_final_np[:margin, :] > rho_cut).any() or 
        (rho_final_np[-margin:, :] > rho_cut).any() or 
        (rho_final_np[:, :margin] > rho_cut).any() or 
        (rho_final_np[:, -margin:] > rho_cut).any()
    )
    
    if touches_wall:
        n_fragments = 2
        max_rho = np.max(rho_final_np)
        neck_density = 0.0
        print(f"  [Eval] Ecm={ecm:.2f} MeV -> nfrag=2 (NO FUSION: Fragments touched the boundary wall!)")
    else:
        n_fragments, max_rho, neck_density = count_fragments(rho_final, rho_cut=rho_cut)
        print(f"  [Eval] Ecm={ecm:.2f} MeV -> nfrag={n_fragments} (max_rho={max_rho:.3f}, neck_density={neck_density:.3f})")
        
    return n_fragments, trajectory

def find_barrier(model, m, std, initial_density, ecm_min, ecm_max, b, n_step, device, tolerance=0.5, max_iter=30, rho_cut=0.4):
    history = []
    
    print(f"\n--- Starting Barrier Search between {ecm_min} and {ecm_max} MeV ---")
    try:
        nfrag_min, traj_min = evaluate_ecm(model, m, std, initial_density, ecm_min, b, n_step, device, rho_cut)
    except ValueError as e:
        print(f"Error at Ecm_min={ecm_min}: {e}")
        return None, history
        
    nfrag_max, traj_max = evaluate_ecm(model, m, std, initial_density, ecm_max, b, n_step, device, rho_cut)
    
    print(f"Initial Endpoints: Min={ecm_min} (Frags: {nfrag_min}), Max={ecm_max} (Frags: {nfrag_max})")
    
    if nfrag_min == 1 and nfrag_max == 1:
        print(f"Both endpoints give FUSION. The barrier is below {ecm_min:.2f} MeV.")
        return ecm_min, [(ecm_min, nfrag_min), (ecm_max, nfrag_max)]
        
    if nfrag_min >= 2 and nfrag_max >= 2:
        print(f"Both endpoints give SCATTERING. The barrier is above {ecm_max:.2f} MeV.")
        return ecm_max, [(ecm_min, nfrag_min), (ecm_max, nfrag_max)]
        
    if nfrag_min == 1:
        a, b_val = ecm_max, ecm_min
        frag_a, frag_b = nfrag_max, nfrag_min
    else:
        a, b_val = ecm_min, ecm_max
        frag_a, frag_b = nfrag_min, nfrag_max
        
    history = [(a, frag_a), (b_val, frag_b)]
    iteration = 0
    
    while abs(b_val - a) > tolerance and iteration < max_iter:
        midpoint = (a + b_val) / 2.0
        iteration += 1
        
        print(f"Iteration {iteration}: checking midpoint {midpoint:.2f} MeV...")
        try:
            nfrag_mid, traj_mid = evaluate_ecm(model, m, std, initial_density, midpoint, b, n_step, device, rho_cut)
        except ValueError as e:
            # If Coulomb energy too high at midpoint, it acts like a scattering event (or unphysical)
            print(f"  ValueError at {midpoint}: {e}")
            a = midpoint
            history.append((midpoint, -1))
            continue
            
        history.append((midpoint, nfrag_mid))
        if nfrag_mid == 1:
            b_val = midpoint
        elif nfrag_mid >= 2:
            a = midpoint
        else:
            # 0 fragments implies the model diffused the density out of distribution due to high velocity.
            # Physically, this is an excess of energy, so we treat it as an upper bound (Fusion).
            print(f"  Warning: 0 fragments detected at {midpoint} MeV. Assuming upper bound (Fusion).")
            b_val = midpoint
            
    barrier_ecm = (a + b_val) / 2.0
    
    print(f"\n✅ BARRIER BRACKETED: E_cm ≈ {barrier_ecm:.3f} MeV")
    
    # b_val is the closest energy bound that resulted in Fusion (1 fragment)
    print(f"Generating Fusion Animation of the trajectory (using E_cm = {b_val:.3f} MeV)...")
    
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML, display
    
    # Run evaluation exactly at the known fusion energy
    nfrag_final, traj_final = evaluate_ecm(model, m, std, initial_density, b_val, b, n_step, device, rho_cut)
    
    preds_tensor = traj_final.cpu()
    
    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    vmax_p = preds_tensor[:, :, :, 0].max().item()
    vmax_n = preds_tensor[:, :, :, 1].max().item()
    
    im_p = axes[0].imshow(preds_tensor[0, :, :, 0].numpy(), cmap='plasma', vmin=0, vmax=vmax_p)
    im_n = axes[1].imshow(preds_tensor[0, :, :, 1].numpy(), cmap='viridis', vmin=0, vmax=vmax_n)
    
    axes[0].set_title("Protons")
    axes[1].set_title("Neutrons")
    fig.colorbar(im_p, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im_n, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    def update(frame):
        im_p.set_data(preds_tensor[frame, :, :, 0].numpy())
        im_n.set_data(preds_tensor[frame, :, :, 1].numpy())
        fig.suptitle(f"Fusion Animation | E_cm = {b_val:.2f} MeV | Step: {frame}/{n_step}", fontsize=14)
        return [im_p, im_n]
    
    anim_f = animation.FuncAnimation(fig, update, frames=n_step, interval=100, blit=False)
    plt.close(fig)
    
    # Directly display in the Jupyter Notebook!
    display(HTML(anim_f.to_jshtml()))
    
    return barrier_ecm, history
