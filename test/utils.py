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
        x=0.6,
        ha="center",
        va="bottom",
    )
    # ax.set_ylabel("Prediction (W)", fontsize=fontsize, fontweight="bold")
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
    log_scale=False,
):
    sorted_table = latency_table.sort_values(by="Measurement")

    x = sorted_table["Measurement"]

    ax.scatter(
        x,
        sorted_table["Roofline"],
        marker="x",
        alpha=0.4,
        s=15,
        label="Roofline",
        zorder=2,
        linewidths=0.8,
    )
    if has_baseline:
        ax.scatter(
            x,
            sorted_table["Baseline"],
            marker="^",
            alpha=0.4,
            s=15,
            label="LLMCompass",
            zorder=3,
            edgecolors="none",
        )
    ax.scatter(
        x,
        sorted_table["Ours"],
        color="navy",
        alpha=0.4,
        s=15,
        label="Ours",
        zorder=4,
        edgecolors="none",
    )
    ax.plot(x, x, "r--", linewidth=1, label="Ideal (y=x)", zorder=5)

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
        leg.set_zorder(1.5)
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_alpha(0.5)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    if log_scale:
        import math

        from matplotlib.ticker import (
            FixedLocator,
            FuncFormatter,
            LogLocator,
            NullFormatter,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")

        def _nice_ticks(vmin, vmax, subs):
            d0 = math.floor(math.log10(vmin)) - 1
            d1 = math.ceil(math.log10(vmax)) + 1
            ticks = []
            for d in range(d0, d1 + 1):
                for s in subs:
                    v = s * 10**d
                    if vmin <= v <= vmax:
                        ticks.append(v)
            return sorted(set(ticks))

        _fmt = FuncFormatter(lambda v, _: f"{v:g}")
        xmin, xmax = ax.get_xlim()
        ax.xaxis.set_major_locator(FixedLocator(_nice_ticks(xmin, xmax, [1.0])))
        ax.xaxis.set_major_formatter(_fmt)
        ax.xaxis.set_minor_locator(FixedLocator(_nice_ticks(xmin, xmax, [5.0])))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=0)
        ax.xaxis.grid(True, which="minor", linestyle="--", alpha=0.5)
        # y axis: 1/2/5 subdivisions for denser grid
        ax.yaxis.set_major_locator(LogLocator(subs=[1.0, 2.0, 5.0], numticks=15))
        ax.yaxis.set_major_formatter(_fmt)
        ax.yaxis.set_minor_formatter(NullFormatter())


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
        alpha=0.4,
        markeredgewidth=0,
    )
    h_roofline = Line2D(
        [0],
        [0],
        color="C0",
        marker="x",
        linestyle="None",
        markersize=ms,
        alpha=0.3,
        markeredgewidth=0.8,
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
            alpha=0.3,
            markeredgewidth=0,
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
        frameon=True,
    )
    mape_legend.get_frame().set_alpha(0)
    mape_legend.get_frame().set_linewidth(0.8)
    if old_legend is not None:
        ax.add_artist(old_legend)
    # Keep MAPE legend on top of the main legend when they overlap
    if mape_legend is not None:
        mape_legend.set_zorder(old_legend.get_zorder() + 1 if old_legend else 10)
