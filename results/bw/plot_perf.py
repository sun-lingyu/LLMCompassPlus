import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

COLOR_BASE = "#4F89BA"  # no-overlap bars (comm fully serialized)
COLOR_OVERLAP = "#F2C28A"  # overlap bars (ideal comm/compute overlap)
IDEAL_LINE_COLOR = "black"
IDEAL_LINE_STYLE = "--"
IDEAL_LINE_WIDTH = 1.0
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 0.4

# Bandwidth reduction between adjacent (doubling) BW points below which we
# consider the curve to have entered "diminishing returns".
KNEE_THRESHOLD = 0.05

# Compact paper-figure layout: narrow bars packed tightly within each group.
FIGSIZE = (1.2, 1.5)
BAR_GROUP_WIDTH = 0.62
BAR_GAP = 0.02

FONT_AXIS_LABEL = 9
FONT_TICK = 9
FONT_MODEL_NAME = 8
FONT_IDEAL = 8
FONT_KNEE = 8


def _split_data_and_ideal(df, bw_col):
    """Split the raw CSV into (numeric BW rows, 'Ideal' row, 'Ideal single' row)."""
    first = df[bw_col].astype(str).str.strip()
    is_ideal = first.str.lower() == "ideal"
    is_ideal_single = first.str.lower() == "ideal single"

    ideal_df = df.loc[is_ideal]
    ideal_single_df = df.loc[is_ideal_single]

    plot_df = df.loc[~is_ideal & ~is_ideal_single].copy()
    plot_df[bw_col] = pd.to_numeric(plot_df[bw_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[bw_col])
    plot_df = plot_df.sort_values(bw_col).reset_index(drop=True)

    ideal_row = ideal_df.iloc[0] if len(ideal_df) else None
    ideal_single_row = ideal_single_df.iloc[0] if len(ideal_single_df) else None
    return plot_df, ideal_row, ideal_single_row


def _find_base_overlap_columns(series_columns):
    """Identify the (base_col, overlap_col, model_name) triple.

    Expects one column named "<model>" and a sibling "<model> overlap".
    """
    overlap_col = None
    model_name = None
    for c in series_columns:
        s = str(c).strip()
        if re.search(r"\boverlap$", s, flags=re.IGNORECASE):
            overlap_col = c
            model_name = re.sub(r"\s+overlap$", "", s, flags=re.IGNORECASE).strip()
            break

    base_col = None
    if model_name is not None:
        for c in series_columns:
            if c != overlap_col and str(c).strip() == model_name:
                base_col = c
                break
    if base_col is None:
        for c in series_columns:
            if c != overlap_col:
                base_col = c
                break
    if model_name is None and base_col is not None:
        model_name = str(base_col).strip()

    return base_col, overlap_col, model_name


def _find_knee(bw, lat, threshold=KNEE_THRESHOLD):
    """Find the first BW point beyond which further doubling of bandwidth
    only reduces latency by less than `threshold` (diminishing returns).

    Returns (index, found) where `found` is False if the reduction never
    drops below the threshold within the sampled range (falls back to the
    last point).
    """
    n = len(lat)
    for i in range(n - 1):
        reduction = (lat[i] - lat[i + 1]) / lat[i]
        if reduction < threshold:
            return i, True
    return n - 1, False


def _resolve_bw_csv_paths(bw_dir, args):
    if args.spec_tokens == "-1":
        prefix = (
            f"{args.device}_{args.precision}_{args.phase}_{args.seq_len}_"
            f"{args.parallelism}_perf"
        )
    else:
        prefix = (
            f"{args.device}_{args.precision}_{args.phase}_{args.seq_len}_"
            f"{args.spec_tokens}_{args.parallelism}_perf"
        )

    return (
        os.path.join(bw_dir, f"{prefix}.csv"),
        os.path.join(bw_dir, f"{prefix}.png"),
    )


def _build_series(base_col, overlap_col):
    series = [("No overlap", base_col, COLOR_BASE)]
    if overlap_col is not None:
        series.append(("Overlap", overlap_col, COLOR_OVERLAP))
    return series


def _plot_bars(ax, x, bw, plot_df, series, ideal_row, ideal_single_row, bar_width):
    """Draw the speedup bars for every series and collect Ideal/knee info."""
    n_series = len(series)
    ideal_speedups = []
    knee_infos = []
    max_speedup = 0.0

    for idx, (label, col, color) in enumerate(series):
        lat = plot_df[col].astype(float).to_numpy()
        ideal_single = float(ideal_single_row[col])
        speedup = ideal_single / lat
        max_speedup = max(max_speedup, float(speedup.max()))

        offset = (idx - (n_series - 1) / 2) * (bar_width + BAR_GAP)
        ax.bar(
            x + offset,
            speedup,
            width=bar_width,
            color=color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            zorder=2,
            label=label,
        )

        if ideal_row is not None:
            ideal_lat = float(ideal_row[col])
            ideal_sp = ideal_single / ideal_lat
            ideal_speedups.append((label, ideal_sp, color))
            max_speedup = max(max_speedup, ideal_sp)

        knee_idx, found = _find_knee(bw, lat)
        knee_infos.append(
            {
                "label": label,
                "color": color,
                "bw": bw[knee_idx],
                "speedup": float(speedup[knee_idx]),
                "x": x[knee_idx] + offset,
                "found": found,
            }
        )

    return ideal_speedups, knee_infos, max_speedup


def _draw_ideal_lines(ax, ideal_speedups, anchor_left=True):
    """Draw one dashed horizontal line per distinct ideal-speedup value.

    Series whose ideal speedup values coincide (within 0.1%) share a single
    line/label so we don't stack redundant annotations. The line/label are
    always drawn in `IDEAL_LINE_COLOR` (regardless of series). `anchor_left`
    chooses which side of the axes the label sits on; callers pick whichever
    side is less likely to be crossed by a knee-annotation arrow.
    """
    values = []
    for _label, value, _color in ideal_speedups:
        if not any(abs(value - v) / max(abs(v), 1e-9) < 1e-3 for v in values):
            values.append(value)

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    x_frac = 0.02 if anchor_left else 0.98
    ha = "left" if anchor_left else "right"
    for value in values:
        ax.axhline(
            y=value,
            color=IDEAL_LINE_COLOR,
            linestyle=IDEAL_LINE_STYLE,
            linewidth=IDEAL_LINE_WIDTH,
            zorder=1,
        )
        ax.text(
            x_frac,
            value,
            f"{value:.1f}\u00d7",
            transform=trans,
            ha=ha,
            va="bottom",
            fontsize=FONT_IDEAL,
            fontweight="bold",
            color=IDEAL_LINE_COLOR,
            zorder=4,
            bbox={
                "boxstyle": "round,pad=0.06",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )


def _annotate_knees(ax, knee_infos, ymax):
    """Annotate each series' diminishing-returns point.

    Each series gets its own fixed horizontal "rail" near the top of the
    axes (rather than an offset relative to its own bar height), so labels
    never collide with the bars or the Ideal line regardless of the
    underlying data. When two rails would still land visually close in x
    (adjacent/same BW index), the labels are additionally nudged apart
    horizontally.
    """
    n = len(knee_infos)
    row_fracs = [0.90] if n == 1 else list(np.linspace(0.80, 0.97, n))

    text_x = [info["x"] for info in knee_infos]
    if n == 2 and abs(text_x[1] - text_x[0]) < 0.9:
        mid = sum(text_x) / 2
        spread = 0.5 if text_x[1] >= text_x[0] else -0.5
        text_x = [mid - spread, mid + spread]

    for info, tx, row_frac in zip(knee_infos, text_x, row_fracs):
        star = "" if info["found"] else "*"
        # The x position already sits on the corresponding BW tick, so the
        # label only needs the speedup value to stay compact.
        text = f"{info['speedup']:.2f}\u00d7{star}"
        ax.annotate(
            text,
            xy=(info["x"], info["speedup"]),
            xytext=(tx, row_frac * ymax),
            ha="center",
            va="center",
            fontsize=FONT_KNEE,
            fontweight="bold",
            color=info["color"],
            zorder=5,
            arrowprops={
                "arrowstyle": "-",
                "color": info["color"],
                "lw": 0.6,
                "shrinkA": 0,
                "shrinkB": 5,
            },
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )
        if not info["found"]:
            print(
                f"  [warn] '{info['label']}': latency reduction never dropped "
                f"below {KNEE_THRESHOLD * 100:.0f}% within the sampled BW "
                f"range; annotated the last point instead (marked with *)."
            )


def _save_bw_legend(out_path):
    handles = [
        Patch(facecolor=COLOR_BASE, edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH),
        Patch(
            facecolor=COLOR_OVERLAP, edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH
        ),
        Line2D(
            [0],
            [0],
            color=IDEAL_LINE_COLOR,
            linestyle=IDEAL_LINE_STYLE,
            linewidth=IDEAL_LINE_WIDTH,
        ),
    ]
    labels = ["Serialized", "Optimistic overlap", "Zero-cost communication"]

    fig_leg, ax_leg = plt.subplots(figsize=(len(handles) * 1.1, 0.45))
    ax_leg.set_axis_off()
    leg = ax_leg.legend(
        handles,
        labels,
        loc="center",
        ncol=len(handles),
        fontsize=9,
        frameon=True,
        fancybox=False,
        edgecolor="#888888",
        facecolor="white",
        labelspacing=0.3,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    leg.get_frame().set_linewidth(0.8)
    fig_leg.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig_leg)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot speedup-vs-Ideal-Single bar chart across interconnect "
            "bandwidths (results/bw CSV)"
        )
    )
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
        "--parallelism",
        type=str,
        required=True,
        help="Parallelism degree in filename (e.g., 4 for TP/CP width)",
    )
    parser.add_argument(
        "--spec_tokens",
        type=str,
        required=False,
        default="-1",
        help="Number of special tokens (e.g., 64); -1 to omit from filename",
    )
    parser.add_argument(
        "--bw_dir",
        type=str,
        default=None,
        help="Directory containing CSV (default: this script's directory)",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Save standalone horizontal legend figure",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bw_dir = args.bw_dir if args.bw_dir else script_dir

    input_filename, out_filename = _resolve_bw_csv_paths(bw_dir, args)

    df = pd.read_csv(input_filename)
    if df.shape[1] < 2:
        raise ValueError(
            "CSV must have bandwidth column plus at least one series column"
        )

    bw_col = df.columns[0]
    series_cols = list(df.columns[1:])
    base_col, overlap_col, _model_name = _find_base_overlap_columns(series_cols)
    if base_col is None:
        raise ValueError("Could not find a base (non-overlap) series column")

    plot_df, ideal_row, ideal_single_row = _split_data_and_ideal(df, bw_col)
    if plot_df.empty:
        raise ValueError("CSV has no numeric bandwidth rows to plot")
    if ideal_single_row is None:
        raise ValueError("CSV is missing the 'Ideal single' baseline row")

    bw = plot_df[bw_col].astype(float).to_numpy()
    x = np.arange(len(bw))

    series = _build_series(base_col, overlap_col)
    n_series = len(series)
    bar_width = (BAR_GROUP_WIDTH - BAR_GAP * (n_series - 1)) / n_series

    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ideal_speedups, knee_infos, max_speedup = _plot_bars(
        ax, x, bw, plot_df, series, ideal_row, ideal_single_row, bar_width
    )

    ymax = max_speedup * 1.55
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.65, len(x) - 1 + 0.65)

    xticklabels = [f"{int(v)}" if float(v).is_integer() else f"{v:g}" for v in bw]
    ax.set_xticks(x)
    # Narrow figures can't fit all (up to 3-digit) BW labels side-by-side at
    # fontsize>=9 without them colliding; rotate them vertically instead of
    # shrinking the font or dropping labels.
    narrow = FIGSIZE[0] < 2.0
    if narrow:
        ax.set_xticklabels(xticklabels, rotation=90, ha="center", va="top")
    else:
        ax.set_xticklabels(xticklabels)
    ax.set_xlabel(
        "BW (GB/s)", fontsize=FONT_AXIS_LABEL, fontweight="bold", labelpad=1.5
    )
    ax.set_ylabel("Speedup", fontsize=FONT_AXIS_LABEL, fontweight="bold", labelpad=1.5)
    ax.tick_params(axis="both", labelsize=FONT_TICK, length=2, pad=1.5)

    # Anchor the Ideal-line label on whichever side the knee arrows are
    # less likely to cross (arrows run ~vertically from each knee's bar).
    avg_knee_x = np.mean([info["x"] for info in knee_infos]) if knee_infos else 0.0
    anchor_left = avg_knee_x >= (len(x) - 1) / 2
    _draw_ideal_lines(ax, ideal_speedups, anchor_left=anchor_left)
    _annotate_knees(ax, knee_infos, ymax)

    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="--",
        color="#B8B8B8",
        linewidth=0.5,
        alpha=0.95,
        zorder=0,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.7)

    bottom = 0.34 if FIGSIZE[0] < 2.0 else 0.24
    fig.subplots_adjust(left=0.32, right=0.97, top=0.97, bottom=bottom)
    plt.savefig(out_filename, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Chart successfully saved to: {out_filename}")

    if args.legend:
        leg_filename = os.path.join(script_dir, "bw_legend.png")
        _save_bw_legend(leg_filename)
        print(f"Legend saved to: {leg_filename}")


if __name__ == "__main__":
    main()
