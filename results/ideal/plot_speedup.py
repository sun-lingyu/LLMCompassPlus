import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

COLOR_TP = "#4F89BA"
COLOR_CP = "#D67F42"
LINE_TP = "#A3C8EF"
LINE_CP = "#F2C28A"
MARKER_EDGEWIDTH = 0.85

SEQ_LENS = [512, 768, 1024, 1280, 1536]
MARKERS = ["o", "s", "^", "D", "v"]
LINE_STYLES = ["-", "--", ":"]


def _find_csv(script_dir, device, precision, phase, seq_len):
    """Locate the perf CSV for given parameters, tolerating spaces in filenames."""
    stem = f"{device}_{precision}_{phase}_{seq_len}_perf"
    for fname in os.listdir(script_dir):
        if not fname.endswith(".csv"):
            continue
        clean = fname.replace(" ", "")
        if clean == f"{stem}.csv":
            return os.path.join(script_dir, fname)
    return None


def _load_speedup(csv_path, parallelism, degree):
    """Return {model: speedup} for the given parallelism type and degree."""
    if csv_path is None or not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    strat_col = df.columns[0]
    models = list(df.columns[1:])

    baseline_mask = df[strat_col].astype(str).str.strip() == "No Parallelism"
    if not baseline_mask.any():
        return {}
    baseline = df.loc[baseline_mask].iloc[0]

    target_label = f"{parallelism}={degree}"
    target_mask = df[strat_col].astype(str).str.strip() == target_label
    if not target_mask.any():
        return {}
    target = df.loc[target_mask].iloc[0]

    result = {}
    for model in models:
        try:
            base_val = float(baseline[model])
            tgt_val = float(target[model])
            if tgt_val > 0:
                result[model] = base_val / tgt_val
        except (ValueError, TypeError):
            pass
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Plot speedup vs seq_len for TP/CP parallelism"
    )
    parser.add_argument(
        "--device", type=str, required=True, help="Device name (e.g., Thor, Orin)"
    )
    parser.add_argument(
        "--precision", type=str, required=True, help="Precision (e.g., fp8, int8)"
    )
    parser.add_argument(
        "--phase", type=str, required=True, help="Phase (e.g., prefill, decode)"
    )
    parser.add_argument(
        "--degree", type=int, required=True, help="Parallelism degree (e.g., 2, 4)"
    )
    parser.add_argument("--legend", action="store_true", help="Show legend")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    include_cp = True
    parallelisms = ["TP", "CP"] if include_cp else ["TP"]

    # Collect speedup data per parallelism type and model across all seq_lens
    data = {p: {} for p in parallelisms}
    models = None

    for seq_len in SEQ_LENS:
        csv_path = _find_csv(
            script_dir, args.device, args.precision, args.phase, seq_len
        )
        for p in parallelisms:
            speedups = _load_speedup(csv_path, p, args.degree)
            if models is None and speedups:
                models = list(speedups.keys())
            for model in models or []:
                data[p].setdefault(model, []).append(speedups.get(model, np.nan))

    if models is None:
        raise ValueError(
            f"No data found for device={args.device}, precision={args.precision}, "
            f"phase={args.phase}, degree={args.degree}."
        )

    color_map = {"TP": COLOR_TP, "CP": COLOR_CP}
    line_color_map = {"TP": LINE_TP, "CP": LINE_CP}

    # Y-axis range and grid step depend on degree
    if args.degree == 4:
        y_min, y_max, y_step = 1.0, 4.0, 0.6
    else:  # degree == 2
        y_min, y_max, y_step = 1.0, 2.0, 0.2

    def _short_model_name(model):
        """Strip common prefixes like 'Qwen3_', keep e.g. '1.7B'."""
        return re.sub(r"^[A-Za-z0-9]+_", "", str(model).strip())

    def _build_legend_handles(parallelisms, models, color_map, line_color_map):
        """One entry per (model, parallelism) combination: e.g. '1.7B TP'."""
        handles, labels = [], []
        for p in parallelisms:
            pt_color = color_map[p]
            ln_color = line_color_map[p]
            for i, model in enumerate(models):
                h = Line2D(
                    [0],
                    [0],
                    color=ln_color,
                    linewidth=1.5,
                    linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                    marker=MARKERS[i % len(MARKERS)],
                    markersize=3,
                    markerfacecolor=pt_color,
                    markeredgecolor=pt_color,
                    markeredgewidth=MARKER_EDGEWIDTH,
                )
                handles.append(h)
                labels.append(f"{_short_model_name(model)} {p}")
        return handles, labels

    def _draw_lines(ax):
        for p in parallelisms:
            pt_color = color_map[p]
            ln_color = line_color_map[p]
            for i, model in enumerate(models):
                y = data[p][model]
                ax.plot(
                    SEQ_LENS,
                    y,
                    color=ln_color,
                    linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                    linewidth=1.2,
                    marker=MARKERS[i % len(MARKERS)],
                    markersize=3,
                    markerfacecolor=pt_color,
                    markeredgecolor=pt_color,
                    markeredgewidth=MARKER_EDGEWIDTH,
                    zorder=2,
                )

    def _style_ax(ax, show_xlabel=True):
        yticks = np.arange(y_min, y_max + y_step * 0.5, y_step)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(yticks)
        ax.set_xticks(SEQ_LENS)
        ax.set_xticklabels([str(s) for s in SEQ_LENS], fontsize=8)
        ax.tick_params(axis="y", labelsize=9)
        if show_xlabel:
            ax.set_xlabel("Sequence length", fontsize=7, fontweight="bold", labelpad=1)
        ax.set_ylabel("Speedup", fontsize=9, fontweight="bold")
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

    # --- Main chart ---
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(2.3, 1.2))
    _draw_lines(ax)
    _style_ax(ax)

    out_stem = (
        f"{args.device}_{args.precision}_{args.phase}_degree{args.degree}_speedup"
    )
    out_path = os.path.join(script_dir, f"{out_stem}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor="white")
    print(f"Chart successfully saved to: {out_path}")

    # --- Standalone legend figure ---
    handles, labels = _build_legend_handles(
        parallelisms, models, color_map, line_color_map
    )
    fig_leg = plt.figure(figsize=(1.1, 1.5))
    ax_leg = fig_leg.add_axes([0, 0, 1, 1])
    ax_leg.set_axis_off()
    leg = ax_leg.legend(
        handles,
        labels,
        fontsize=8,
        loc="center",
        frameon=True,
        fancybox=False,
        edgecolor="#888888",
        facecolor="white",
        labelspacing=0.3,
        handlelength=1.8,
        handletextpad=0.4,
        borderpad=0.5,
    )
    leg.get_frame().set_linewidth(0.8)
    leg_path = os.path.join(script_dir, f"{out_stem}_legend.png")
    fig_leg.savefig(leg_path, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"Legend saved to: {leg_path}")


if __name__ == "__main__":
    main()
