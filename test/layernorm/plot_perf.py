import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

file_dir = os.path.dirname(os.path.abspath(__file__))

color_machine = sns.color_palette("flare", 1)
color_simulate = sns.color_palette("Blues_d", 4)[1:]


def plot_latency(latency_table, ax, title, is_first=False, is_last=False):
    sorted_table = latency_table.sort_values(by="Measured")

    x = sorted_table["Measured"]

    ax.scatter(x, sorted_table["Ours"], color="navy", alpha=0.6, s=8, label="Ours")
    ax.scatter(
        x, sorted_table["Roofline"], marker="x", alpha=0.6, s=8, label="Roofline"
    )
    ax.plot(x, x, "r--", linewidth=1, label="Ideal (y=x)")

    ax.set_title(title)
    ax.set_xlabel("Measured Latency (ms)", fontsize=6)

    if is_first:
        ax.set_ylabel("Predicted Latency (ms)")
    if is_last:
        ax.legend(loc="best")

    ax.grid(True, linestyle="--", alpha=0.5)


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, choices=["Orin", "Thor"])
parser.add_argument(
    "--mode",
    type=str,
    choices=["prefill", "decode"],
)
parser.add_argument("--precision", type=str, choices=["fp16"])
args = parser.parse_args()

try:
    plt.style.use("seaborn-v0_8")
except:
    plt.style.use("ggplot")

# Plot one figure
fig, axes = plt.subplots(1, 1, figsize=(3, 2.8), sharey=True)
latency_table = pd.read_csv(f"{file_dir}/results_perf/{args.device}_{args.mode}.csv")
plot_latency(latency_table, axes, title=args.mode, is_first=True, is_last=True)

mape_ours = (
    np.mean(
        np.abs(
            (latency_table["Ours"] - latency_table["Measured"])
            / latency_table["Measured"]
        )
    )
    * 100
)
mape_roofline = (
    np.mean(
        np.abs(
            (latency_table["Roofline"] - latency_table["Measured"])
            / latency_table["Measured"]
        )
    )
    * 100
)

axes.text(
    0.5,
    0.01,
    f"MAPE: Ours {mape_ours:.1f}%, Roofline {mape_roofline:.1f}%",
    ha="center",
    va="bottom",
    transform=axes.transAxes,
    fontsize=6,
)
fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.1)
fig.savefig(f"{file_dir}/results_perf/{args.device}_{args.mode}.png", dpi=300)
