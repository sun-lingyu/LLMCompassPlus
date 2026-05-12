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
        nargs="+",
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
    parser.add_argument(
        "--log",
        action="store_true",
        help="Use log scale for both axes",
    )
    args = parser.parse_args()

    try:
        plt.style.use("seaborn-v0_8-white")
    except:
        plt.style.use("ggplot")

    # Load and concatenate data from all specified models
    per_model_dfs = []
    for model in args.model:
        csv_files = glob.glob(
            str(
                Path(
                    f"{file_dir}/results_perf/{model}/{args.device}/{args.precision}/{args.mode}"
                )
                / "*.csv"
            )
        )
        if not csv_files:
            print(f"Warning: no CSV files found for model={model}, skipping.")
            continue
        dfs = [pd.read_csv(f) for f in csv_files]
        model_df = pd.concat(dfs, ignore_index=True)
        model_df = model_df.drop_duplicates(
            subset=["Ours", "Baseline", "Roofline"],
            keep="first",
        )
        per_model_dfs.append(model_df)

    latency_table = pd.concat(per_model_dfs, ignore_index=True)
    latency_table = latency_table.sort_values(by="Measurement", ascending=False)

    # Determine output path
    model_tag = "+".join(args.model)
    out_dir = Path(f"{file_dir}/results_perf")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_tag}_{args.device}_{args.precision}_{args.mode}.png"

    has_baseline = args.precision in ("int8", "fp16")

    # Plot one figure
    fig, axes = plt.subplots(1, 1, figsize=(2.5, 2), sharey=True)
    plot_latency(
        latency_table,
        axes,
        title="+".join(args.model),
        precision=args.precision,
        is_first=True,
        is_last=True,
        has_baseline=has_baseline,
        log_scale=args.log,
    )
    add_mape_annotation(
        latency_table,
        axes,
        args.precision,
        has_baseline=has_baseline,
    )
    fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.1)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved to {out_path}")
