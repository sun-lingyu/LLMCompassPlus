import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def _degrees_for_prefix(labels, prefix):
    degs = []
    for s in labels:
        m = re.match(rf"^{prefix}=(\d+)", str(s).strip())
        if m:
            degs.append(int(m.group(1)))
    return sorted(set(degs))


def strategy_color_fn(strategies):
    tp_degrees = _degrees_for_prefix(strategies, "TP")
    cp_degrees = _degrees_for_prefix(strategies, "CP")
    n_tp, n_cp = len(tp_degrees), len(cp_degrees)
    tp_palette = (
        sns.blend_palette(["#A3C8EF", "#4F89BA"], n_colors=n_tp) if n_tp else []
    )
    cp_palette = (
        sns.blend_palette(["#F2C28A", "#D67F42"], n_colors=n_cp) if n_cp else []
    )
    tp_deg_to_i = {d: i for i, d in enumerate(tp_degrees)}
    cp_deg_to_i = {d: i for i, d in enumerate(cp_degrees)}
    baseline_color = "#9A9A9A"

    def color_for(label):
        s = str(label).strip()
        if "No Parallelism" in s:
            return baseline_color
        m = re.match(r"^TP=(\d+)", s)
        if m and tp_deg_to_i:
            return tp_palette[tp_deg_to_i[int(m.group(1))]]
        m = re.match(r"^CP=(\d+)", s)
        if m and cp_deg_to_i:
            return cp_palette[cp_deg_to_i[int(m.group(1))]]
        return baseline_color

    return color_for


def _parse_strategy_row(label):
    """Return (kind, degree) for bar grouping: base / tp / cp / other."""
    s = str(label).strip()
    if "No Parallelism" in s:
        return ("base", None)
    m = re.match(r"^TP=(\d+)", s)
    if m:
        return ("tp", int(m.group(1)))
    m = re.match(r"^CP=(\d+)", s)
    if m:
        return ("cp", int(m.group(1)))
    return ("other", None)


def compute_bar_offsets(strategy_labels, width, gap_normal, gap_zero=0.0):
    """
    One x-offset per row. Adjacent TP=d and CP=d use gap_zero; all other
    consecutive pairs use gap_normal so e.g. TP=2 and TP=4 stay separated
    when no CP rows are present.
    """
    n = len(strategy_labels)
    if n == 0:
        return []
    kinds = [_parse_strategy_row(s) for s in strategy_labels]
    cumulative = 0.0
    offsets = []
    for i in range(n):
        offsets.append(cumulative)
        cumulative += width
        if i < n - 1:
            k1, d1 = kinds[i]
            k2, d2 = kinds[i + 1]
            if k1 == "tp" and k2 == "cp" and d1 is not None and d1 == d2:
                cumulative += gap_zero
            else:
                cumulative += gap_normal
    mean_offset = sum(offsets) / len(offsets)
    return [o - mean_offset for o in offsets]


def main():
    parser = argparse.ArgumentParser(description="Generate Bar Chart for Model Latency")
    parser.add_argument(
        "--device", type=str, required=True, help="Device name (e.g., Orin)"
    )
    parser.add_argument(
        "--precision", type=str, required=True, help="Precision (e.g., fp16)"
    )
    parser.add_argument(
        "--phase", type=str, required=True, help="Phase (e.g., prefill)"
    )
    parser.add_argument(
        "--seq_len", type=str, required=True, help="Sequence length (e.g., 1024)"
    )
    parser.add_argument(
        "--spec_tokens",
        type=str,
        required=False,
        default="-1",
        help="Number of special tokens (e.g., 64)",
    )
    parser.add_argument("--legend", action="store_true", help="Show legend")
    args = parser.parse_args()

    if args.spec_tokens == "-1":
        file_name_prefix = (
            f"{args.device}_{args.precision}_{args.phase}_{args.seq_len}_perf"
        )
    else:
        file_name_prefix = f"{args.device}_{args.precision}_{args.phase}_{args.seq_len}_{args.spec_tokens}_perf"
    input_filename = f"{file_name_prefix}.csv"
    out_filename = f"{file_name_prefix}.png"

    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"Cannot find CSV file: {input_filename}")

    df = pd.read_csv(input_filename)

    models = df.columns[1:]
    strategies = df.iloc[:, 0].values
    strategy_labels = [str(s).strip() for s in strategies]

    baseline_latency = df.iloc[0, 1:].astype(float).values

    plt.style.use("seaborn-v0_8-white")
    color_for_strategy = strategy_color_fn(strategies)

    width = 0.15
    gap_normal = 0.04
    gap_zero = 0.0

    offsets = compute_bar_offsets(strategy_labels, width, gap_normal, gap_zero)

    if args.phase == "prefill":
        fig, ax = plt.subplots(figsize=(4, 1.5))
    else:
        fig, ax = plt.subplots(figsize=(2.5, 1.2))
    x_indices = np.arange(len(models))

    for pos, (_, row) in enumerate(df.iterrows()):
        y_values = row[1:].astype(float).values
        label = strategy_labels[pos]

        bars = ax.bar(
            x_indices + offsets[pos],
            y_values,
            width,
            color=color_for_strategy(label),
            label=label,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        if re.match(r"^CP=", label):
            ann_xytext = (1.5, 0)
        elif re.match(r"^TP=", label) and args.phase == "prefill":
            ann_xytext = (-1.5, 0)
        else:
            ann_xytext = (0, 0)

        for j, bar in enumerate(bars):
            latency = y_values[j]
            speedup = baseline_latency[j] / latency
            ax.annotate(
                f"{speedup:.1f}x",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=ann_xytext,
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
                color="black",
            )

    ax.set_ylabel(df.columns[0], fontsize=9, fontweight="bold")
    ax.set_xticks(x_indices)
    # if args.phase == "decode":
    #     xticklabels = [re.sub(r"^Qwen3_", "", str(m)) for m in models]
    # else:
    #     xticklabels = list(models)
    xticklabels = list(models)
    ax.set_xticklabels(xticklabels, fontsize=8)

    max_y = df.iloc[:, 1:].astype(float).max().max()
    ax.set_ylim(0, max_y * 1.2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Prefill: optional FPS reference lines (latency ms = 1000 / FPS).
    _prefill_fps_by_device = {
        "Orin": (30, 10),
        "Thor": (30,),
    }
    x_position = 0.5 if args.device == "Orin" else 1.1
    if args.phase == "prefill" and args.device in _prefill_fps_by_device:
        y_cap = max_y * 1.2
        for fps in _prefill_fps_by_device[args.device]:
            y_ref = 1000.0 / fps
            if y_ref > y_cap:
                continue
            ax.axhline(
                y=y_ref,
                color="#B22222",
                linestyle="--",
                linewidth=0.9,
                zorder=1,
                alpha=0.95,
            )
            ax.text(
                x_position,
                y_ref + 0.02 * max_y,
                f"{fps} FPS",
                ha="right",
                va="bottom",
                fontsize=6,
                zorder=1,
                color="#B22222",
                fontstyle="italic",
            )

    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="--",
        color="#B8B8B8",
        linewidth=0.7,
        alpha=0.95,
        zorder=0,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)

    if args.legend:
        leg = ax.legend(
            fontsize=7,
            title_fontsize=8,
            loc="upper left",
            frameon=True,
            fancybox=False,
            edgecolor="#888888",
            facecolor="white",
            labelspacing=0.2,
        )
        leg.set_zorder(1)
        leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(out_filename, dpi=300, facecolor="white")
    print(f"Chart successfully saved to: {out_filename}")


if __name__ == "__main__":
    main()
