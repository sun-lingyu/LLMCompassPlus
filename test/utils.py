import base64
import os
import shlex
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
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
        "num_layers": 24,
    },
    "Qwen3_0_6B": {
        "head_dim": 128,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "hidden_act": "silu",
        "num_layers": 28,
    },
    "Qwen3_1_7B": {
        "head_dim": 128,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "hidden_act": "silu",
        "num_layers": 28,
    },
    "Qwen3_4B": {
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "hidden_act": "silu",
        "num_layers": 36,
    },
    "Qwen3_8B": {
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "hidden_act": "silu",
        "num_layers": 36,
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
    y_true,
    y_pred,
    feature_names,
    coefs,
    intercept,
    r2,
    mape,
    save_dir,
    title_suffix="",
    fontsize=13,
):
    try:
        plt.style.use("seaborn-v0_8-white")
    except:
        plt.style.use("ggplot")

    fig, ax = plt.subplots(1, 1, figsize=(2.2, 2.2))

    ax.scatter(y_true, y_pred, color="navy", alpha=0.6, s=15, label="Records")
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal (y=x)"
    )
    fig.suptitle(
        f"$R^2={r2:.2f}, MAPE={mape * 100:.2f}\\%$",
        fontsize=fontsize - 2,
        x=0.5,
        ha="center",
        y=0.98,
    )
    fig.supxlabel(
        "Measurement (W)",
        fontsize=fontsize,
        fontweight="bold",
        x=1.0,
        ha="right",
        va="bottom",
    )
    ax.set_ylabel("Prediction (W)", fontsize=fontsize, fontweight="bold")
    leg = ax.legend(
        loc="upper left",
        fontsize=fontsize - 2,
        borderaxespad=0.4,
        handletextpad=0.2,
        labelspacing=0.3,
        handlelength=1.5,
        frameon=True,
        borderpad=0.2,
    )
    leg.set_zorder(0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", labelsize=fontsize - 1)

    text = ""
    for i, v in enumerate(coefs):
        if v != 0:
            if feature_names[i] == "FMA":
                text += f" FMA = {v * 1e12:.2f} pJ/op\n"
            elif feature_names[i] == "DRAM Access Byte":
                text += f" DRAM R/W = {v * 1e12:.2f} pJ/B\n"
    text += f"Intercept = {intercept:.2f} W"

    plt.figtext(
        0.95,
        0.17,
        text,
        ha="right",
        va="bottom",
        fontsize=fontsize - 3,
    )

    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.1)
    plt.subplots_adjust(bottom=0.17)
    plt.subplots_adjust(top=0.9)
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


def plot_latency(
    latency_table,
    ax,
    title,
    precision,
    is_first=False,
    is_last=False,
    has_baseline=False,
    fontsize=14,
):
    sorted_table = latency_table.sort_values(by="Measurement")

    x = sorted_table["Measurement"]

    ax.scatter(x, sorted_table["Ours"], color="navy", alpha=0.6, s=15, label="Ours")
    ax.scatter(
        x, sorted_table["Roofline"], marker="x", alpha=0.6, s=15, label="Roofline"
    )
    if has_baseline:
        ax.scatter(
            x, sorted_table["Baseline"], marker="^", alpha=0.6, s=15, label="LLMCompass"
        )
    ax.plot(x, x, "r--", linewidth=1, label="Ideal (y=x)")

    # ax.set_title(title)
    fig = ax.figure
    fig.supxlabel(
        "Measurement (ms)",
        fontsize=fontsize,
        fontweight="bold",
        x=1.0,
        ha="right",
        va="bottom",
    )
    ax.tick_params(axis="both", labelsize=fontsize)

    if is_first:
        ax.set_ylabel("Prediction (ms)", fontsize=fontsize, fontweight="bold")
    if is_last:
        leg = ax.legend(
            loc="upper left",
            fontsize=fontsize - 3,
            borderaxespad=0.4,
            handletextpad=0.2,
            labelspacing=0.3,
            handlelength=1.2,
            frameon=True,
            borderpad=0.2,
        )
        leg.set_zorder(0)
        leg.get_frame().set_linewidth(0.8)

    ax.grid(True, linestyle="--", alpha=0.5)


def add_mape_annotation(df, ax, precision, fontsize=11, has_baseline=False):
    def calc_mape(pred, true):
        return np.mean(np.abs((pred - true) / true)) * 100

    mape_ours = calc_mape(df["Ours"], df["Measurement"])
    mape_roofline = calc_mape(df["Roofline"], df["Measurement"])

    # Line2D proxies match plot_latency scatter styles (implicit cycle: Roofline=C0, Baseline=C1)
    ms = max(3.0, fontsize * 0.45)
    h_ours = Line2D(
        [0],
        [0],
        color="navy",
        marker="o",
        linestyle="None",
        markersize=ms,
        alpha=0.6,
    )
    h_roofline = Line2D(
        [0],
        [0],
        color="C0",
        marker="x",
        linestyle="None",
        markersize=ms,
        alpha=0.6,
        markeredgewidth=1.2,
    )
    if has_baseline:
        mape_baseline = calc_mape(df["Baseline"], df["Measurement"])
        h_baseline = Line2D(
            [0],
            [0],
            color="C1",
            marker="^",
            linestyle="None",
            markersize=ms,
            alpha=0.6,
        )
        handles = [h_ours, h_baseline, h_roofline]
        labels = [
            f"{mape_ours:.1f}%",
            f"{mape_baseline:.1f}%",
            f"{mape_roofline:.1f}%",
        ]
    else:
        handles = [h_ours, h_roofline]
        labels = [f"{mape_ours:.1f}%", f"{mape_roofline:.1f}%"]

    old_legend = ax.get_legend()
    mape_legend = ax.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.0),
        borderaxespad=0,
        handletextpad=-0.2,
        labelspacing=0.3,
        fontsize=fontsize,
        title="MAPE:",
        title_fontsize=fontsize,
        frameon=False,
    )
    if old_legend is not None:
        ax.add_artist(old_legend)
    # Keep MAPE legend on top of the main legend when they overlap
    if mape_legend is not None:
        mape_legend.set_zorder(old_legend.get_zorder() + 1 if old_legend else 10)
