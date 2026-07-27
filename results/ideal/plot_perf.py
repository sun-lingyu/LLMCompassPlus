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


# Prefill @ seq_len=1024: borrow InternVision from native-precision CSVs.
_INTERNVISION_PREFILL = {
    ("Orin", "int8"): ("fp16", "InternVision (f16)"),
    ("Thor", "fp4"): ("fp8", "InternVision (f8)"),
}


def _find_csv(script_dir, device, precision, phase, seq_len):
    """Locate perf CSV, tolerating stray spaces in filenames."""
    stem = f"{device}_{precision}_{phase}_{seq_len}_perf"
    for fname in os.listdir(script_dir):
        if not fname.endswith(".csv"):
            continue
        if fname.replace(" ", "") == f"{stem}.csv":
            return os.path.join(script_dir, fname)
    return os.path.join(script_dir, f"{stem}.csv")


def _internvision_source_column(df):
    for col in df.columns[1:]:
        if str(col).strip().startswith("InternVision"):
            return col
    return None


def merge_internvision_prefill(df, device, precision, phase, seq_len, script_dir):
    """Inject InternVision column for mixed-precision prefill bar charts."""
    if phase != "prefill":
        return df
    key = (device, precision)
    if key not in _INTERNVISION_PREFILL:
        return df
    if str(seq_len) != "1024":
        return df

    src_precision, col_label = _INTERNVISION_PREFILL[key]
    if col_label in df.columns:
        return df

    src_path = _find_csv(script_dir, device, src_precision, phase, seq_len)
    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"Cannot find InternVision source CSV: {src_path}"
        )

    src_df = pd.read_csv(src_path)
    src_col = _internvision_source_column(src_df)
    if src_col is None:
        raise ValueError(f"No InternVision column in source CSV: {src_path}")

    strat_col = df.columns[0]
    src_by_strat = src_df.set_index(strat_col)[src_col]
    iv_values = src_by_strat.reindex(df[strat_col].astype(str).str.strip()).values
    if pd.isna(iv_values).any():
        raise ValueError(
            f"Strategy rows mismatch between {src_path} and target prefill CSV."
        )

    merged = df.copy()
    merged.insert(1, col_label, iv_values)
    return merged


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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = _find_csv(
        script_dir, args.device, args.precision, args.phase, args.seq_len
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find CSV file: {csv_path}")

    df = pd.read_csv(csv_path)
    df = merge_internvision_prefill(
        df, args.device, args.precision, args.phase, args.seq_len, script_dir
    )

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

    print(f"\n{'Strategy':<12} | {'Model':<20} | Speedup")
    print("-" * 44)
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
        is_baseline = "No Parallelism" in label
        for j, bar in enumerate(bars):
            if is_baseline:
                continue
            latency = y_values[j]
            speedup = baseline_latency[j] / latency
            print(f"  {label:<12} | {models[j]:<20} | {speedup:.2f}x")

    ax.set_ylabel(df.columns[0], fontsize=10, fontweight="bold")
    ax.set_xticks(x_indices)
    # if args.phase == "decode":
    #     xticklabels = [re.sub(r"^Qwen3_", "", str(m)) for m in models]
    # else:
    #     xticklabels = list(models)
    xticklabels = list(models)
    ax.set_xticklabels(xticklabels, fontsize=8.5)

    max_y = df.iloc[:, 1:].astype(float).max().max()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Prefill: optional FPS reference lines (latency ms = 1000 / FPS).
    _prefill_fps_by_device = {
        "Orin": (30, 10),
        "Thor": (30,),
    }
    x_position = 0.2 if args.device == "Orin" else 1.1
    if args.phase == "prefill" and args.device in _prefill_fps_by_device:
        fps_y_refs = [1000.0 / fps for fps in _prefill_fps_by_device[args.device]]
        max_fps_ref = max(fps_y_refs)
        current_top = ax.get_ylim()[1]
        if max_fps_ref >= current_top * 0.95:
            ax.set_ylim(0, max_fps_ref * 1.2)
        for fps in _prefill_fps_by_device[args.device]:
            y_ref = 1000.0 / fps
            ax.axhline(
                y=y_ref,
                color="#333333",
                linestyle="--",
                linewidth=0.9,
                zorder=1,
                alpha=0.95,
            )
            ax.text(
                x_position,
                y_ref + 0.005 * max_y,
                f"{fps} FPS",
                ha="right",
                va="bottom",
                fontsize=10,
                zorder=1,
                color="#333333",
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

    plt.tight_layout()
    plt.savefig(out_filename, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Chart successfully saved to: {out_filename}")

    if args.legend:
        handles, labels = ax.get_legend_handles_labels()
        n = len(handles)
        fig_leg, ax_leg = plt.subplots(figsize=(n * 1.3, 0.45))
        ax_leg.set_axis_off()
        leg = ax_leg.legend(
            handles,
            labels,
            loc="center",
            ncol=n,
            fontsize=10,
            frameon=True,
            fancybox=False,
            edgecolor="#888888",
            facecolor="white",
            labelspacing=0.3,
            handlelength=1.5,
            columnspacing=1.0,
        )
        leg.get_frame().set_linewidth(0.8)
        leg_filename = "perf_legend.png"
        fig_leg.savefig(leg_filename, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig_leg)
        print(f"Legend saved to: {leg_filename}")


if __name__ == "__main__":
    main()
