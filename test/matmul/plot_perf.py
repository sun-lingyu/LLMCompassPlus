import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import glob
from pathlib import Path

file_dir = os.path.dirname(os.path.abspath(__file__))

color_machine = sns.color_palette("flare", 1)
color_simulate = sns.color_palette("Blues_d", 4)[1:]

def plot_latency(
    latency_table,
    ax,
    title,
    precision,
    is_first = False,
    is_last = False
):
    x = [_ for _ in range(latency_table.shape[0])]

    ax.plot(x, latency_table["Ours"], marker="x", markersize=4, color=color_simulate[2], label="Ours")
    if precision != "int4":
        ax.plot(x, latency_table["Baseline"], marker="o", markersize=4, color=color_simulate[1], label="LLMCompass")
    ax.plot(x, latency_table["Roofline"], marker="^", markersize=4, color=color_simulate[0], label="Roofline")
    ax.plot(x, latency_table["CUTLASS"], marker=" ", linestyle="--", linewidth=1.5, color=color_machine[0], label="Measurement")
    
    ax.set_title(title)
    ax.set_xticklabels([])
    if is_first:
        ax.set_ylabel("Latency (ms)")
    if is_last:
        ax.legend(loc="best")
    ax.grid(True)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["prefill", "decode"],)
parser.add_argument("--model", type=str, choices=["InternVision", "Qwen3_0_6B", "Qwen3_1_7B", "Qwen3_4B", "Qwen3_8B"])
parser.add_argument("--precision", type=str, choices=["fp16", "int8", "int4"])
parser.add_argument("--plot_one_figure", action='store_true')
args = parser.parse_args()

if args.plot_one_figure:
    fig, axes = plt.subplots(1, 1, figsize=(3, 2.8), sharey=True)
    csv_files = glob.glob(str(Path(f"{file_dir}/results_perf/{args.model}/{args.precision}/{args.mode}") / "*.csv"))
    dfs = [pd.read_csv(f) for f in csv_files]
    latency_table = pd.concat(dfs, ignore_index=True)
    latency_table = latency_table.drop_duplicates(
        subset=["Ours", "Baseline", "Roofline"],
        keep="first",
    )
    latency_table = latency_table.sort_values(
        by="CUTLASS",
        ascending=False
    )
    plot_latency(latency_table, axes, title=args.model, precision=args.precision, is_first=True, is_last=True)

    mape_ours = np.mean(np.abs((latency_table['Ours'] - latency_table['CUTLASS']) / latency_table['CUTLASS'])) * 100
    mape_baseline = np.mean(np.abs((latency_table['Baseline'] - latency_table['CUTLASS']) / latency_table['CUTLASS'])) * 100
    mape_roofline = np.mean(np.abs((latency_table['Roofline'] - latency_table['CUTLASS']) / latency_table['CUTLASS'])) * 100

    if args.precision != "int4":
        axes.text(0.5, -0.08, f"MAPE: Ours {mape_ours:.1f}%, LLMCompass {mape_baseline:.1f}%, Roofline {mape_roofline:.1f}%", ha='center', va='bottom', transform=axes.transAxes, fontsize=6)
    else:
        axes.text(0.5, -0.08, f"MAPE: Ours {mape_ours:.1f}%, Roofline {mape_roofline:.1f}%", ha='center', va='bottom', transform=axes.transAxes, fontsize=6)
    fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.1)
    fig.savefig(f"{file_dir}/results_perf/{args.model}/{args.precision}/{args.mode}/{args.mode}.png", dpi=300)
else:
    fig, axes = plt.subplots(1, 4, figsize=(2 * 4, 3), sharey=True)
    if args.mode == "prefill":
        fig_1, axes_1 = plt.subplots(1, 4, figsize=(2 * 4, 3), sharey=True)

    for idx, test in enumerate(["qkv_proj", "o_proj", "up_proj", "down_proj"]):
        if args.mode == "prefill":
            latency_table = pd.read_csv(f"{file_dir}/results_perf/{args.model}/{args.precision}/{args.mode}/{test}_CP.csv")
            plot_latency(latency_table, axes_1[idx], title=test, precision=args.precision, is_first=(idx==0), is_last=(idx==3))
        
        latency_table = pd.read_csv(f"{file_dir}/results_perf/{args.model}/{args.precision}/{args.mode}/{test}_TP.csv")
        plot_latency(latency_table, axes[idx], title=test, precision=args.precision, is_first=(idx==0), is_last=(idx==3))

    fig.tight_layout()
    fig.savefig(f"{file_dir}/results_perf/{args.model}/{args.precision}/{args.mode}/TP.png", dpi=300)
    if args.mode == "prefill":
        fig_1.tight_layout()
        fig_1.savefig(f"{file_dir}/results_perf/{args.model}/{args.precision}/{args.mode}/CP.png", dpi=300)