import argparse
import os

import numpy as np

from test.utils import fit_single_rail, plot_fitting_results, print_rail_results

file_dir = os.path.dirname(os.path.abspath(__file__))

# MEM rail idle power (静态功耗), same across matmul/flashattn
mem_intercept_dict = {"Orin": 0.5, "Thor": 6.7}
MATMUL_PRECISIONS = ["fp16", "int8", "int4", "fp8", "fp4"]
FLASHATTN_PRECISIONS = ["fp16"]
LAYERNORM_PRECISIONS = ["fp16"]


def load_matmul_data(device):
    matmul_dir = os.path.join(file_dir, "matmul")
    dram_rates = []
    y_mem_list = []
    for precision in MATMUL_PRECISIONS:
        cache_path = f"{matmul_dir}/temp/power_features_cache_{precision}.{device}.npz"
        if not os.path.exists(cache_path):
            print(f"[matmul/{precision}] Cache not found, skipping: {cache_path}")
            continue
        try:
            data = np.load(cache_path)
            X = data["X"]  # shape (N, 5): FMA, Input, Output, L2, DRAM
            y_mem = data["y_mem"]  # shape (N,)
        except Exception as e:
            print(f"[matmul/{precision}] Failed to load cache: {e}, skipping.")
            continue
        dram_col = X[:, -1]  # DRAM Access Byte rate, index 4
        dram_rates.extend(dram_col.tolist())
        y_mem_list.extend(y_mem.tolist())
        print(f"[matmul/{precision}] Loaded {len(dram_col)} records from cache.")
    return dram_rates, y_mem_list


def load_flashattn_data(device):
    flashattn_dir = os.path.join(file_dir, "flashattn")
    dram_rates = []
    y_mem_list = []
    for precision in FLASHATTN_PRECISIONS:
        cache_path = (
            f"{flashattn_dir}/temp/power_features_cache_{precision}.{device}.npz"
        )
        if not os.path.exists(cache_path):
            print(f"[flashattn/{precision}] Cache not found, skipping: {cache_path}")
            continue
        try:
            data = np.load(cache_path)
            X = data["X"]  # shape (N, 4): FMA, Input, Output, DRAM
            y_mem = data["y_mem"]  # shape (N,)
        except Exception as e:
            print(f"[flashattn/{precision}] Failed to load cache: {e}, skipping.")
            continue
        dram_col = X[:, -1]  # DRAM Access Byte rate, index 3
        dram_rates.extend(dram_col.tolist())
        y_mem_list.extend(y_mem.tolist())
        print(f"[flashattn/{precision}] Loaded {len(dram_col)} records from cache.")
    return dram_rates, y_mem_list


def load_layernorm_data(device):
    layernorm_dir = os.path.join(file_dir, "layernorm")
    dram_rates = []
    y_mem_list = []
    for precision in LAYERNORM_PRECISIONS:
        cache_path = (
            f"{layernorm_dir}/temp/power_features_cache_{precision}.{device}.npz"
        )
        if not os.path.exists(cache_path):
            print(f"[layernorm/{precision}] Cache not found, skipping: {cache_path}")
            continue
        try:
            data = np.load(cache_path)
            X = data["X"]  # shape (N, 1): DRAM
            y_mem = data["y_mem"]  # shape (N,)
        except Exception as e:
            print(f"[layernorm/{precision}] Failed to load cache: {e}, skipping.")
            continue
        dram_col = X[:, -1]  # DRAM Access Byte rate, index 0
        dram_rates.extend(dram_col.tolist())
        y_mem_list.extend(y_mem.tolist())
        print(f"[layernorm/{precision}] Loaded {len(dram_col)} records from cache.")
    return dram_rates, y_mem_list


def fit_mem_power(device):
    print(f"\n{'=' * 80}")
    print(f" UNIFIED MEM POWER FITTING  (device={device})")
    print(f"{'=' * 80}")
    print("\n[Step 1] Loading Matmul data...")
    matmul_dram, matmul_y_mem = load_matmul_data(device)
    print(f"  -> {len(matmul_dram)} records loaded.")
    print("\n[Step 2] Loading FlashAttn data...")
    flashattn_dram, flashattn_y_mem = load_flashattn_data(device)
    print(f"  -> {len(flashattn_dram)} records loaded.")
    print("\n[Step 3] Loading LayerNorm data...")
    layernorm_dram, layernorm_y_mem = load_layernorm_data(device)
    print(f"  -> {len(layernorm_dram)} records loaded.")
    all_dram = matmul_dram + flashattn_dram + layernorm_dram
    all_y_mem = matmul_y_mem + flashattn_y_mem + layernorm_y_mem
    if len(all_dram) == 0:
        print("No data available. Exiting.")
        return
    print(
        f"\n[Step 3] Fitting with {len(all_dram)} total records"
        f" (matmul: {len(matmul_dram)}, flashattn: {len(flashattn_dram)}, layernorm: {len(layernorm_dram)})"
    )
    X = np.array(all_dram).reshape(-1, 1)
    y_mem = np.array(all_y_mem)
    mem_intercept = mem_intercept_dict[device]
    feature_names = ["DRAM Access Byte"]
    feat_map = {"DRAM Access Byte": 0}
    features_to_use = ["DRAM Access Byte"]
    res_mem = fit_single_rail(
        X,
        y_mem - mem_intercept,
        features_to_use,
        "Unified MEM",
        feat_map,
        feature_names,
        enforce_positive=True,
        fit_intercept=False,
    )
    print_rail_results("Mem", res_mem, feature_names, features_to_use)
    print(f"\n[Info] Static MEM Power (intercept_dict) = {mem_intercept:.4f} W")
    print(
        f"[Info] MEM Power = {res_mem['coefs'][0]:.6e} * DRAM_rate"
        f" + {mem_intercept:.4f} W"
    )
    results_dir = os.path.join(file_dir, "results_power")
    os.makedirs(results_dir, exist_ok=True)
    plot_fitting_results(
        y_mem,
        res_mem["y_pred"] + mem_intercept,
        feature_names,
        res_mem["coefs"],
        res_mem["intercept"] + mem_intercept,
        res_mem["r2"],
        res_mem["mape"],
        results_dir,
        title_suffix=f"mem_unified_{device}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit unified MEM power model across all ops and precisions."
    )
    parser.add_argument("device", type=str, choices=["Orin", "Thor"])
    args = parser.parse_args()
    fit_mem_power(args.device)
