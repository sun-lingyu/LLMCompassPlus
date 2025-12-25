import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict

file_dir = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE_TEMPLATE = f"{file_dir}/power_features_cache"

def plot_fitting_results(y_true, y_pred, feature_names, coefs, intercept, r2, mse, title_suffix=""):
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('ggplot')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, color='navy', alpha=0.6, s=60, label='Records')
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')
    ax1.set_title(f'Physical Power Model (NNLS)\n$R^2={r2:.4f}, MSE={mse:.4f}$', fontsize=14)
    ax1.set_xlabel('Measured Power (W)', fontsize=12)
    ax1.set_ylabel('Predicted Power (W)', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = axes[1]
    y_pos = np.arange(len(feature_names))
    
    bars = ax2.barh(y_pos, coefs, color='forestgreen', alpha=0.8, edgecolor='k')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feature_names, fontsize=12)
    ax2.set_xlabel('Energy Cost (Joules / Op or Byte)', fontsize=12)
    ax2.set_title('Estimated Energy Per Operation (Must be >= 0)', fontsize=14)
    
    for i, v in enumerate(coefs):
        ax2.text(v, i, f' {v:.2e} J', va='center', fontsize=10, fontweight='bold')

    plt.figtext(0.5, 0.02, f"Static Power (Intercept) = {intercept:.4f} W", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    save_path = f"{file_dir}/power_nnls_fitting_{title_suffix}.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[Info] Plot saved to: {save_path}")

def load_or_generate_data(args):
    cache_path = f"{CACHE_FILE_TEMPLATE}_{args.precision}.npz"
    X, y_soc, y_mem = None, None, None

    if os.path.exists(cache_path):
        print(f"Found cache file: {cache_path}")
        try:
            data = np.load(cache_path)
            if 'y_soc' in data and 'y_mem' in data:
                X = data['X']
                y_soc = data['y_soc']
                y_mem = data['y_mem']
                print(f"Loaded {len(X)} records from cache.")
                return X, y_soc, y_mem
        except Exception as e:
            print(f"Error loading cache: {e}. Will re-calculate.")

    print("Generating features from simulation...")
    pcb = device_dict["Orin"]
    existing_data = []
    
    json_path = f"{file_dir}/cutlass_power_log.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            content = f.read().strip()
            if content:
                existing_data = json.loads(content)
    else:
        raise FileNotFoundError(f"{json_path} not found")

    X_features_raw = []
    X_features = []
    y_soc_list = []
    y_mem_list = []
    M_list = []

    for record in existing_data:
        M, N, K, precision = record['M'], record['N'], record['K'], record['precision']
        # if not (M ==1024 and N == 4096 and K == 3072):
        #     continue
        if precision == "fp16":
            act_dt, wei_dt, int_dt = data_type_dict["fp16"], data_type_dict["fp16"], data_type_dict["fp32"]
        elif precision == "int8":
            act_dt, wei_dt, int_dt = data_type_dict["int8"], data_type_dict["int8"], data_type_dict["int32"]
        elif precision == "int4":
            act_dt, wei_dt, int_dt = data_type_dict["fp16"], data_type_dict["int4"], data_type_dict["fp32"]
        else:
            continue
        
        if precision != args.precision:
            continue

        model = Matmul(activation_data_type=act_dt, weight_data_type=wei_dt, intermediate_data_type=int_dt)
        _ = model(Tensor([M, K], act_dt), Tensor([K, N], wei_dt))
        
        latency_ms = 1000 * (model.compile_and_simulate(pcb, compile_mode="heuristic-GPU") + 2773 / pcb.compute_module.clock_freq)
        runtime_s = latency_ms / 1000.0

        features = [
            model.systolic_array_fma_count / runtime_s, # 0: FMA
            model.reg_access_count / runtime_s,         # 1: Reg
            model.l1_access_size / runtime_s,           # 2: L1
            model.l2_access_size / runtime_s,           # 3: L2
            model.mem_access_size / runtime_s           # 4: DRAM
        ]
        
        X_features_raw.append([model.systolic_array_fma_count, 
                               model.reg_access_count, 
                               model.l1_access_size, 
                               model.l2_access_size, 
                               model.mem_access_size])
        X_features.append(features)
        y_soc_list.append(record['power_VDD_GPU_SOC'])
        y_mem_list.append(record['power_VDDQ_VDD2_1V8AO'])
        M_list.append(M)
        
        print(f"M={M}, N={N}, K={K} | Latency={latency_ms:.2f}ms | SOC={record['power_VDD_GPU_SOC']}W, MEM={record['power_VDDQ_VDD2_1V8AO']}W")
        print(f"  Features_raw: FMA={model.systolic_array_fma_count}, Reg={model.reg_access_count}, L1={model.l1_access_size}, L2={model.l2_access_size}, DRAM={model.mem_access_size}")

    if len(X_features) > 0:
        X = np.array(X_features)
        y_soc = np.array(y_soc_list)
        y_mem = np.array(y_mem_list)
        
        print(f"Saving new cache to: {cache_path}")
        np.savez(cache_path, X=X, y_soc=y_soc, y_mem=y_mem)
    else:
        print("No valid data generated.")
        return None, None, None
    prefill_X_raw = [X[idx][4] for idx in range(len(X)) if M_list[idx] >= 256]
    prefill_y_mem = [y_mem[idx] for idx in range(len(X)) if M_list[idx] >= 256]
    decode_X_raw = [X[idx][4] for idx in range(len(X)) if M_list[idx] < 256]
    decode_y_mem = [y_mem[idx] for idx in range(len(X)) if M_list[idx] < 256]
    plt.scatter(prefill_X_raw, prefill_y_mem, color='blue', alpha=0.6, s=60, label='prefill')
    plt.scatter(decode_X_raw, decode_y_mem, color='yellow', alpha=0.6, s=60, label='decode')
    plt.xlabel('DRAM access size (Bytes/s)')
    plt.ylabel('Power VDDQ_VDD2_1V8AO (W)')
    plt.legend()
    plt.savefig(f"{file_dir}/soc_scatter.png", dpi=300)

    return X, y_soc, y_mem

def fit_and_analyze_rails(X_raw, y_soc, y_mem, args):
    full_feature_names = ["SysArr FMA", "Reg Access", "L1 Access", "L2 Access", "DRAM Access"]
    
    feat_map = {name: i for i, name in enumerate(full_feature_names)}

    # Custom features here
    # ==============================================================================
    soc_features_to_use = ["SysArr FMA", "Reg Access", "L1 Access", "L2 Access"] 

    mem_features_to_use = ["DRAM Access"]
    # ==============================================================================

    print("\n" + "="*80)
    print(" DUAL RAIL POWER MODELING (Configurable Feature Subsets) ")
    print("="*80)

    def fit_single_rail(X_full, y, features_to_use, rail_label, enforce_positive=True, fit_intercept=True):
        print(f"\n--- [{rail_label}] Fitting with: {features_to_use} ---")
        
        try:
            indices = [feat_map[name] for name in features_to_use]
        except KeyError as e:
            raise ValueError(f"Feature name {e} not found in full_feature_names")

        X_subset = X_full[:, indices]

        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X_subset)

        model = LinearRegression(positive=enforce_positive, fit_intercept=fit_intercept)
        model.fit(X_scaled, y)

        subset_coefs = model.coef_ / scaler.scale_
        intercept = model.intercept_

        aligned_coefs = np.zeros(len(full_feature_names))
        for idx_in_subset, original_idx in enumerate(indices):
            aligned_coefs[original_idx] = subset_coefs[idx_in_subset]

        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        return {
            "coefs": aligned_coefs,
            "intercept": intercept,
            "y_pred": y_pred,
            "r2": r2,
            "mse": mse,
            "model": model
        }

    res_soc = fit_single_rail(X_raw, y_soc, soc_features_to_use, 
                              "Rail 1: VDD_GPU_SOC", enforce_positive=True)
    res_mem = fit_single_rail(X_raw, y_mem, mem_features_to_use, 
                              "Rail 2: VDDQ_VDD2_1V8AO", enforce_positive=True, fit_intercept=False)

    print("\n" + "-"*85)
    print(f" SoC RESULTS (R^2: {res_soc['r2']:.4f}, MSE: {res_soc['mse']:.4f})")
    print(f" SoC Static Power: {res_soc['intercept']:.4f} W")
    print("-" * 85)
    print(f"{'Component':<15} | {'Coef (J/op)':<20} | {'Status'}")
    print("-" * 85)
    for i, name in enumerate(full_feature_names):
        val = res_soc['coefs'][i]
        if name in soc_features_to_use:
            status = "Fitted" if abs(val) > 1e-15 else "Zeroed by Solver"
        else:
            status = "Ignored (Config)"
        
        print(f"{name:<15} | {val:.6e}           | {status}")

    print("\n" + "-"*85)
    print(f" Mem RESULTS (R^2: {res_mem['r2']:.4f}, MSE: {res_mem['mse']:.4f})")
    print(f" Mem Static Power: {res_mem['intercept']:.4f} W")
    print("-" * 85)
    print(f"{'Component':<15} | {'Coef (J/op)':<20} | {'Status'}")
    print("-" * 85)
    for i, name in enumerate(full_feature_names):
        val = res_mem['coefs'][i]
        if name in mem_features_to_use:
            status = "Fitted" if abs(val) > 1e-15 else "Zeroed by Solver"
        else:
            status = "Ignored (Config)"
        
        print(f"{name:<15} | {val:.6e}           | {status}")

    plot_fitting_results(y_soc, res_soc['y_pred'], full_feature_names, 
                         res_soc['coefs'], res_soc['intercept'], 
                         res_soc['r2'], res_soc['mse'], title_suffix=f"soc_{args.precision}")
                         
    plot_fitting_results(y_mem, res_mem['y_pred'], full_feature_names, 
                         res_mem['coefs'], res_mem['intercept'], 
                         res_mem['r2'], res_mem['mse'], title_suffix=f"mem_{args.precision}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("precision", type=str, choices=["fp16", "int8", "int4"])
    args = parser.parse_args()

    X, y_soc, y_mem = load_or_generate_data(args)

    if X is not None and len(X) > 0:
        fit_and_analyze_rails(X, y_soc, y_mem, args)
    else:
        print("Exiting: No data available.")