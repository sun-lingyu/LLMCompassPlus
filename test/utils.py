import base64
import os
import shlex
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler

file_dir = os.path.dirname(os.path.abspath(__file__))


test_model_dict = {
    "InternVision": {
        "head_dim": 64,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
    },
    "Qwen3_0_6B": {
        "head_dim": 128,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "hidden_act": "silu",
    },
    "Qwen3_1_7B": {
        "head_dim": 128,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "hidden_act": "silu",
    },
    "Qwen3_4B": {
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "hidden_act": "silu",
    },
    "Qwen3_8B": {
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "hidden_act": "silu",
    },
}


def run_remote_command(user, host, port, remote_cmd, work_dir=None):
    remote_cmd_str = " ".join(shlex.quote(arg) for arg in remote_cmd)
    if work_dir is not None:
        remote_cmd_str = f"cd {work_dir} && {remote_cmd_str}"
    target = f"{user}@{host}" if user is not None else host
    ssh_cmd = ["ssh", "-p", str(port), target, remote_cmd_str]
    proc = subprocess.run(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(
            f"ssh/remote_cmd exited with code {proc.returncode}\n"
            f"SSH Command: {' '.join(ssh_cmd)}\n"
            f"Output:\n{output}"
        )
    return output


def run_power_monitor(
    full_cmd, total_duration, valid_duration, device, user, host, port
):
    print(
        f"Measuring power for {total_duration}s and take {valid_duration}s in the middle..."
    )
    print(full_cmd)

    power_monitor_path = os.path.abspath(os.path.join(file_dir, "power_monitor.py"))
    if not os.path.exists(power_monitor_path):
        raise FileNotFoundError(f"Cannot find power_monitor at {power_monitor_path}")

    with open(power_monitor_path, "r", encoding="utf-8") as f:
        script_body = f.read()

    variables_header = f"""
FULL_CMD = {repr(full_cmd)}
VALID_START_TIME = {total_duration / 2 - valid_duration / 2}
VALID_DURATION = {valid_duration}
DEVICE = "{device}"
"""
    remote_script_source = variables_header + "\n" + script_body
    b64_script = base64.b64encode(remote_script_source.encode("utf-8")).decode("utf-8")
    shell_pipeline = f"echo {b64_script} | base64 -d | python3"
    remote_cmd = ["bash", "-c", shell_pipeline]
    output = run_remote_command(user, host, port, remote_cmd).strip()

    try:
        avg_power_GPU = float(output.splitlines()[-2])
        avg_power_MEM = float(output.splitlines()[-1])
    except ValueError:
        raise RuntimeError(f"Could not parse power output. Received:\n{output}")

    return avg_power_GPU, avg_power_MEM


def plot_fitting_results(
    y_true, y_pred, feature_names, coefs, intercept, r2, mape, save_dir, title_suffix=""
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
        f"Physical Power Model (NNLS)\n$R^2={r2:.4f}, MAPE={mape * 100:.2f}\\%$",
        fontsize=14,
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
    save_path = f"{save_dir}/power_nnls_fitting_{title_suffix}.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[Info] Plot saved to: {save_path}")


def fit_single_rail(
    X_full,
    y,
    features_to_use,
    rail_label,
    feat_map,
    full_feature_names,
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


def print_rail_results(label, res, full_feature_names, features_to_use):
    print("\n" + "-" * 85)
    print(f" {label} RESULTS (R^2: {res['r2']:.4f}, MAPE: {res['mape']:.4f})")
    print(f" {label} Static Power: {res['intercept']:.4f} W")
    print("-" * 85)
    print(f"{'Component':<15} | {'Coef (J/op)':<20} | {'Status'}")
    print("-" * 85)
    for i, name in enumerate(full_feature_names):
        val = res["coefs"][i]
        if name in features_to_use:
            status = "Fitted" if abs(val) > 1e-15 else "Zeroed by Solver"
        else:
            status = "Ignored (Config)"
        print(f"{name:<15} | {val:.6e}           | {status}")
