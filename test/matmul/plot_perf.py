import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

file_dir = os.path.dirname(os.path.abspath(__file__))

color_machine = sns.color_palette("flare", 1)
color_simulate = sns.color_palette("Blues_d", 4)[1:]


def plot_latency(latency_table, ax, title, precision, is_first=False, is_last=False):
    sorted_table = latency_table.sort_values(by="Measurement")

    x = sorted_table["Measurement"]

    ax.scatter(x, sorted_table["Ours"], color="navy", alpha=0.6, s=8, label="Ours")
    ax.scatter(
        x, sorted_table["Roofline"], marker="x", alpha=0.6, s=8, label="Roofline"
    )
    if precision in ("int8", "fp16"):
        ax.scatter(
            x, sorted_table["Baseline"], marker="^", alpha=0.6, s=8, label="LLMCompass"
        )
    ax.plot(x, x, "r--", linewidth=1, label="Ideal (y=x)")

    ax.set_title(title)
    ax.set_xlabel("Measured Latency (ms)", fontsize=6)

    if is_first:
        ax.set_ylabel("Predicted Latency (ms)")
    if is_last:
        ax.legend(loc="best")

    ax.grid(True, linestyle="--", alpha=0.5)


def add_mape_annotation(df, ax, precision, fontsize=6):
    def calc_mape(pred, true):
        return np.mean(np.abs((pred - true) / true)) * 100

    mape_ours = calc_mape(df["Ours"], df["Measurement"])
    mape_baseline = calc_mape(df["Baseline"], df["Measurement"])
    mape_roofline = calc_mape(df["Roofline"], df["Measurement"])

    if precision in ("int8", "fp16"):
        text_str = f"MAPE: Ours {mape_ours:.1f}%, LLMCompass {mape_baseline:.1f}%, Roofline {mape_roofline:.1f}%"
    else:
        text_str = f"MAPE: Ours {mape_ours:.1f}%, Roofline {mape_roofline:.1f}%"

    ax.text(
        0.5,
        0.01,
        text_str,
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=fontsize,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prefill", "decode"],
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["InternVision", "Qwen3_0_6B", "Qwen3_1_7B", "Qwen3_4B", "Qwen3_8B"],
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["Orin", "Thor"],
    )
    parser.add_argument(
        "--precision", type=str, choices=["fp16", "int8", "int4", "fp8", "fp4"]
    )
    args = parser.parse_args()

    try:
        plt.style.use("seaborn-v0_8")
    except:
        plt.style.use("ggplot")

    # Plot one figure
    fig, axes = plt.subplots(1, 1, figsize=(3, 2.8), sharey=True)
    csv_files = glob.glob(
        str(
            Path(
                f"{file_dir}/results_perf/{args.model}/{args.device}/{args.precision}/{args.mode}"
            )
            / "*.csv"
        )
    )
    dfs = [pd.read_csv(f) for f in csv_files]
    latency_table = pd.concat(dfs, ignore_index=True)
    latency_table = latency_table.drop_duplicates(
        subset=["Ours", "Baseline", "Roofline"],
        keep="first",
    )
    latency_table = latency_table.sort_values(by="Measurement", ascending=False)
    plot_latency(
        latency_table,
        axes,
        title=args.model,
        precision=args.precision,
        is_first=True,
        is_last=True,
    )
    add_mape_annotation(latency_table, axes, args.precision)

    fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.1)
    fig.savefig(
        f"{file_dir}/results_perf/{args.model}/{args.device}/{args.precision}/{args.mode}.png",
        dpi=300,
    )
