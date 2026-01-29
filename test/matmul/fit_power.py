import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler

from hardware_model.device import device_dict
from software_model.matmul import Matmul
from software_model.utils import Tensor, data_type_dict
from test.matmul.utils import get_model_shape, get_output_dtype

file_dir = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE_TEMPLATE = f"{file_dir}/temp/power_features_cache"

intercept_dict = {"Orin": {"soc": 25, "mem": 0.5}, "Thor": {"soc": 25, "mem": 6.7}}


def plot_fitting_results(
    y_true, y_pred, feature_names, coefs, intercept, r2, mape, title_suffix=""
):
    try:
        plt.style.use("seaborn-v0_8")
    except:
        plt.style.use("ggplot")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, color="navy", alpha=0.6, s=60, label="Records")
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot(
        [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal (y=x)"
    )
    ax1.set_title(
        f"Physical Power Model (NNLS)\n$R^2={r2:.4f}, MAPE={mape:.4f}$", fontsize=14
    )
    ax1.set_xlabel("Measured Power (W)", fontsize=12)
    ax1.set_ylabel("Predicted Power (W)", fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = axes[1]
    y_pos = np.arange(len(feature_names))

    bars = ax2.barh(y_pos, coefs, color="forestgreen", alpha=0.8, edgecolor="k")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feature_names, fontsize=12)
    ax2.set_xlabel("Energy Cost (Joules / Op or Byte)", fontsize=12)
    ax2.set_title("Estimated Energy Per Operation (Must be >= 0)", fontsize=14)

    for i, v in enumerate(coefs):
        ax2.text(v, i, f" {v:.2e} J", va="center", fontsize=10, fontweight="bold")

    plt.figtext(
        0.5,
        0.02,
        f"Static Power (Intercept) = {intercept:.4f} W",
        ha="center",
        fontsize=12,
        bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5},
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    save_path = f"{file_dir}/results_power/power_nnls_fitting_{title_suffix}.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[Info] Plot saved to: {save_path}")


def load_or_generate_data(args):
    cache_path = f"{CACHE_FILE_TEMPLATE}_{args.precision}.npz"
    X, y_soc, y_mem = None, None, None

    if os.path.exists(cache_path):
        print(f"Found cache file: {cache_path}")
        try:
            data = np.load(cache_path)
            if "y_soc" in data and "y_mem" in data:
                X = data["X"]
                y_soc = data["y_soc"]
                y_mem = data["y_mem"]
                print(f"Loaded {len(X)} records from cache.")
                return X, y_soc, y_mem
        except Exception as e:
            print(f"Error loading cache: {e}. Will re-calculate.")

    print("Generating features from simulation...")
    pcb = device_dict[args.device]
    existing_data = []

    json_path = f"{file_dir}/temp/power_log.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
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

    shape_to_op = {}
    for model in ["InternVision", "Qwen3_1_7B", "Qwen3_4B", "Qwen3_8B"]:
        K_shapes, N_shapes = get_model_shape(model)
        for op_name in K_shapes.keys():
            k = K_shapes[op_name]
            n = N_shapes[op_name]
            shape_to_op[(1024, n, k)] = (op_name, model)
            shape_to_op[(512, n, k)] = (op_name, model)
            shape_to_op[(256, n, k)] = (op_name, model)
            shape_to_op[(128, n, k)] = (op_name, model)
            shape_to_op[(64, n, k)] = (op_name, model)
            shape_to_op[(32, n, k)] = (op_name, model)
            if op_name == "qkv_proj" or op_name == "up_proj":
                shape_to_op[(1024, n // 2, k)] = (op_name, model)
                shape_to_op[(1024, n // 4, k)] = (op_name, model)
                shape_to_op[(128, n // 2, k)] = (op_name, model)
                shape_to_op[(128, n // 4, k)] = (op_name, model)
                shape_to_op[(64, n // 2, k)] = (op_name, model)
                shape_to_op[(64, n // 4, k)] = (op_name, model)
                shape_to_op[(32, n // 2, k)] = (op_name, model)
                shape_to_op[(32, n // 4, k)] = (op_name, model)
            else:  # o_proj or down_proj
                shape_to_op[(1024, n, k // 2)] = (op_name, model)
                shape_to_op[(1024, n, k // 4)] = (op_name, model)
                shape_to_op[(128, n, k // 2)] = (op_name, model)
                shape_to_op[(128, n, k // 4)] = (op_name, model)
                shape_to_op[(64, n, k // 2)] = (op_name, model)
                shape_to_op[(64, n, k // 4)] = (op_name, model)
                shape_to_op[(32, n, k // 2)] = (op_name, model)
                shape_to_op[(32, n, k // 4)] = (op_name, model)

    for record in existing_data:
        M, N, K, precision = record["M"], record["N"], record["K"], record["precision"]
        if precision == "fp16":
            act_dt, wei_dt, int_dt = (
                data_type_dict["fp16"],
                data_type_dict["fp16"],
                data_type_dict["fp32"],
            )
        elif precision == "int8":
            act_dt, wei_dt, int_dt = (
                data_type_dict["int8"],
                data_type_dict["int8"],
                data_type_dict["int32"],
            )
        elif precision == "int4":
            act_dt, wei_dt, int_dt = (
                data_type_dict["fp16"],
                data_type_dict["int4"],
                data_type_dict["fp32"],
            )
        elif precision == "fp8":
            act_dt, wei_dt, int_dt = (
                data_type_dict["fp8"],
                data_type_dict["fp8"],
                data_type_dict["fp32"],
            )
        elif precision == "fp4":
            act_dt, wei_dt, int_dt = (
                data_type_dict["fp4"],
                data_type_dict["fp4"],
                data_type_dict["fp32"],
            )
        else:
            continue
        op_name, model = shape_to_op[(M, N, K)]
        if model == "Qwen3_0_6B":
            continue
        o_dt = get_output_dtype(act_dt, op_name, True)

        if precision != args.precision:
            continue
        model = Matmul(act_dt, wei_dt, int_dt, o_dt, device=args.device)
        _ = model(Tensor([M, K], act_dt), Tensor([K, N], wei_dt))

        latency_ms = 1000 * (
            model.compile_and_simulate(pcb) + pcb.compute_module.launch_latency.matmul
        )
        runtime_s = latency_ms / 1000.0

        features = [
            model.fma_count / runtime_s,  # 0: FMA
            (
                model.M * model.K * model.activation_dtype.word_size
                + model.K * model.N * model.weight_dtype.word_size
            )
            / runtime_s,  # 1: Input
            model.M * model.N * model.output_dtype.word_size / runtime_s,  # 2. Output
            model.l2_access_size / runtime_s,  # 3. l2 cache access size
            model.mem_access_size / runtime_s,  # 4: DRAM
        ]

        X_features_raw.append([model.fma_count, model.mem_access_size])
        X_features.append(features)
        y_soc_list.append(record["power_VDD_GPU_SOC"])
        y_mem_list.append(record["power_VDDQ_VDD2_1V8AO"])
        M_list.append(M)

        print(
            f"M={M}, N={N}, K={K} | Latency={latency_ms:.2f}ms | SOC={record['power_VDD_GPU_SOC']}W, MEM={record['power_VDDQ_VDD2_1V8AO']}W"
        )
        print(f"  Features_raw: FMA={model.fma_count}, DRAM={model.mem_access_size}")

    if len(X_features) > 0:
        X = np.array(X_features)
        y_soc = np.array(y_soc_list)
        y_mem = np.array(y_mem_list)

        print(f"Saving new cache to: {cache_path}")
        np.savez(cache_path, X=X, y_soc=y_soc, y_mem=y_mem)
    else:
        print("No valid data generated.")
        return None, None, None

    return X, y_soc, y_mem


def fit_and_analyze_rails(X_raw, y_soc, y_mem, args):
    full_feature_names = [
        "FMA",
        "INPUT Size",
        "OUTPUT Size",
        "L2 Access Byte",
        "DRAM Access Byte",
    ]
    # full_feature_names = ["FMA", "DRAM Access Byte"]

    feat_map = {name: i for i, name in enumerate(full_feature_names)}

    # Custom features here
    # ==============================================================================
    soc_features_to_use = ["FMA"]

    mem_features_to_use = ["DRAM Access Byte"]
    # ==============================================================================

    print("\n" + "=" * 80)
    print(" DUAL RAIL POWER MODELING (Configurable Feature Subsets) ")
    print("=" * 80)

    def fit_single_rail(
        X_full,
        y,
        features_to_use,
        rail_label,
        enforce_positive=True,
        fit_intercept=True,
    ):
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
        mape = mean_absolute_percentage_error(y, y_pred)

        return {
            "coefs": aligned_coefs,
            "intercept": intercept,
            "y_pred": y_pred,
            "r2": r2,
            "mape": mape,
            "model": model,
        }

    res_soc = fit_single_rail(
        X_raw, y_soc, soc_features_to_use, "Rail 1: VDD_GPU_SOC", enforce_positive=True
    )
    res_mem = fit_single_rail(
        X_raw,
        y_mem - intercept_dict[args.device]["mem"],
        mem_features_to_use,
        "Rail 2: VDDQ_VDD2_1V8AO",
        enforce_positive=True,
        fit_intercept=False,
    )

    print("\n" + "-" * 85)
    print(f" SoC RESULTS (R^2: {res_soc['r2']:.4f}, MAPE: {res_soc['mape']:.4f})")
    print(f" SoC Static Power: {res_soc['intercept']:.4f} W")
    print("-" * 85)
    print(f"{'Component':<15} | {'Coef (J/op)':<20} | {'Status'}")
    print("-" * 85)
    for i, name in enumerate(full_feature_names):
        val = res_soc["coefs"][i]
        if name in soc_features_to_use:
            status = "Fitted" if abs(val) > 1e-15 else "Zeroed by Solver"
        else:
            status = "Ignored (Config)"

        print(f"{name:<15} | {val:.6e}           | {status}")

    print("\n" + "-" * 85)
    print(f" Mem RESULTS (R^2: {res_mem['r2']:.4f}, MAPE: {res_mem['mape']:.4f})")
    print(f" Mem Static Power: {res_mem['intercept']:.4f} W")
    print("-" * 85)
    print(f"{'Component':<15} | {'Coef (J/op)':<20} | {'Status'}")
    print("-" * 85)
    for i, name in enumerate(full_feature_names):
        val = res_mem["coefs"][i]
        if name in mem_features_to_use:
            status = "Fitted" if abs(val) > 1e-15 else "Zeroed by Solver"
        else:
            status = "Ignored (Config)"

        print(f"{name:<15} | {val:.6e}           | {status}")

    plot_fitting_results(
        y_soc,
        res_soc["y_pred"],
        full_feature_names,
        res_soc["coefs"],
        res_soc["intercept"],
        res_soc["r2"],
        res_soc["mape"],
        title_suffix=f"soc_{args.precision}",
    )

    plot_fitting_results(
        y_mem,
        res_mem["y_pred"] + intercept_dict[args.device]["mem"],
        full_feature_names,
        res_mem["coefs"],
        res_mem["intercept"] + intercept_dict[args.device]["mem"],
        res_mem["r2"],
        res_mem["mape"],
        title_suffix=f"mem_{args.precision}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, choices=["Orin", "Thor"])
    parser.add_argument(
        "precision", type=str, choices=["fp16", "int8", "int4", "fp8", "fp4"]
    )
    args = parser.parse_args()

    X, y_soc, y_mem = load_or_generate_data(args)

    if X is not None and len(X) > 0:
        fit_and_analyze_rails(X, y_soc, y_mem, args)
    else:
        print("Exiting: No data available.")
