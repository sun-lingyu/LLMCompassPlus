import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from test.utils import add_mape_annotation, plot_latency

file_dir = os.path.dirname(os.path.abspath(__file__))

color_machine = sns.color_palette("flare", 1)
color_simulate = sns.color_palette("Blues_d", 4)[1:]


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
        subset=["Ours", "Roofline"],
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
