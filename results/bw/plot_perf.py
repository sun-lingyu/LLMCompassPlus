import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

COLOR_TP = "#4F89BA"
COLOR_CP = "#D67F42"
LINE_TP = "#A3C8EF"
LINE_CP = "#F2C28A"
MARKER_EDGEWIDTH = 0.85
IDEAL_LINE_TP = "#4F89BA"
IDEAL_LINE_CP = "#D67F42"


def _parse_column_name(name):
    s = str(name).strip()
    m = re.match(r"^(.+?)\s+(TP|CP)$", s)
    if m:
        return m.group(1).strip(), m.group(2)
    return s, None


def _group_series_by_model(series_columns):
    groups = []
    key_index = {}
    ungrouped = []

    for c in series_columns:
        model, kind = _parse_column_name(c)
        if kind not in ("TP", "CP"):
            ungrouped.append(c)
            continue
        if model not in key_index:
            key_index[model] = len(groups)
            groups.append((model, {"TP": None, "CP": None}))
        groups[key_index[model]][1][kind] = c

    return groups, ungrouped


def _split_data_and_ideal(df, bw_col):
    first = df[bw_col].astype(str).str.strip()
    is_ideal = first.str.lower() == "ideal"
    ideal_df = df.loc[is_ideal]
    plot_df = df.loc[~is_ideal].copy()
    plot_df[bw_col] = pd.to_numeric(plot_df[bw_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[bw_col])

    ideal_row = ideal_df.iloc[0] if len(ideal_df) else None
    return plot_df, ideal_row


def _plot_ideal_hlines(ax, ideal_row, series_cols, include_cp):
    if ideal_row is None:
        return
    for col in series_cols:
        model, kind = _parse_column_name(col)
        if kind == "CP" and not include_cp:
            continue
        if kind == "TP":
            color = IDEAL_LINE_TP
        elif kind == "CP":
            color = IDEAL_LINE_CP
        else:
            continue
        try:
            y = float(ideal_row[col])
        except (TypeError, ValueError, KeyError):
            continue
        if not np.isfinite(y):
            continue
        ax.axhline(
            y=y,
            color=color,
            linestyle="--",
            linewidth=1.25,
            zorder=1,
            clip_on=True,
        )


def _configure_bw_axes(
    ax,
    x,
    df_plot,
    ylim_cols,
    ideal_row,
    series_cols,
    include_cp,
    phase,
    df_nc_plot=None,
    nc_ideal_row=None,
    nc_ylim_cols=None,
):
    ax.set_xscale("log", base=2)
    x_ticks = np.unique(np.sort(x.astype(float)))
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _p: (f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:g}")
        )
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel("ICNT BW (GB/s)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=9, fontweight="bold")

    xmin, xmax = float(x_ticks.min()), float(x_ticks.max())
    ax.set_xlim(xmin / 2**0.25, xmax * 2**0.25)

    ymax = df_plot[ylim_cols].astype(float).to_numpy().max() if ylim_cols else 1.0
    if df_nc_plot is not None and nc_ylim_cols:
        ymax = max(
            ymax,
            float(df_nc_plot[nc_ylim_cols].astype(float).to_numpy().max()),
        )
    if ideal_row is not None:
        for col in ylim_cols:
            try:
                v = float(ideal_row[col])
                if np.isfinite(v):
                    ymax = max(ymax, v)
            except (TypeError, ValueError, KeyError):
                pass
    if nc_ideal_row is not None and nc_ylim_cols:
        for col in nc_ylim_cols:
            try:
                v = float(nc_ideal_row[col])
                if np.isfinite(v):
                    ymax = max(ymax, v)
            except (TypeError, ValueError, KeyError):
                pass
    if phase == "prefill":
        ax.set_ylim(0, ymax * 1.12)
    else:
        ax.set_ylim(0, ymax * 1.35)

    _plot_ideal_hlines(ax, ideal_row, series_cols, include_cp)


def _plot_bw_data_series(
    ax, x, df_plot, model_groups, ungrouped_cols, include_cp, df_nc_plot=None
):
    for _model, cols in model_groups:
        series_styles = (("TP", COLOR_TP, LINE_TP),)
        if include_cp:
            series_styles = (
                ("TP", COLOR_TP, LINE_TP),
                ("CP", COLOR_CP, LINE_CP),
            )
        for _kind, pt_color, ln_color in series_styles:
            col = cols[_kind]
            if col is None:
                continue
            y = df_plot[col].astype(float).values
            ax.plot(
                x,
                y,
                color=ln_color,
                linestyle="-",
                linewidth=1.8,
                marker="o",
                markersize=3,
                markerfacecolor=pt_color,
                markeredgecolor=pt_color,
                markeredgewidth=MARKER_EDGEWIDTH,
                zorder=2,
            )
        if df_nc_plot is not None:
            col = cols.get("TP")
            if col is not None and col in df_nc_plot.columns:
                y_nc = df_nc_plot[col].astype(float).values
                ax.plot(
                    x,
                    y_nc,
                    color=LINE_TP,
                    linestyle="--",
                    linewidth=1.2,
                    marker="x",
                    markersize=3,
                    markerfacecolor=COLOR_TP,
                    markeredgecolor=COLOR_TP,
                    markeredgewidth=MARKER_EDGEWIDTH,
                    zorder=2,
                )

    for col in ungrouped_cols:
        y = df_plot[col].astype(float).values
        ax.plot(
            x,
            y,
            color="#C8C8C8",
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markersize=3,
            markerfacecolor="#4A4A4A",
            markeredgecolor="#4A4A4A",
            markeredgewidth=MARKER_EDGEWIDTH,
            zorder=2,
        )


def _is_thor_device(device):
    return str(device).strip().lower() == "thor"


def _series_cols_for_ylim(model_groups, ungrouped_cols, include_cp):
    cols = []
    for _m, g in model_groups:
        if g.get("TP"):
            cols.append(g["TP"])
        if include_cp and g.get("CP"):
            cols.append(g["CP"])
    cols.extend(ungrouped_cols)
    return cols


def _tp_cp_legend_handles():
    tp_h = Line2D(
        [0],
        [0],
        color=LINE_TP,
        linewidth=2,
        linestyle="-",
        marker="o",
        markersize=3,
        markerfacecolor=COLOR_TP,
        markeredgecolor=COLOR_TP,
        markeredgewidth=MARKER_EDGEWIDTH,
    )
    cp_h = Line2D(
        [0],
        [0],
        color=LINE_CP,
        linewidth=2,
        linestyle="-",
        marker="o",
        markersize=3,
        markerfacecolor=COLOR_CP,
        markeredgecolor=COLOR_CP,
        markeredgewidth=MARKER_EDGEWIDTH,
    )
    return tp_h, cp_h


def _tp_nocontention_legend_handle():
    return Line2D(
        [0],
        [0],
        color=LINE_TP,
        linewidth=1.2,
        linestyle="--",
        marker="x",
        markersize=3,
        markerfacecolor=COLOR_TP,
        markeredgecolor=COLOR_TP,
        markeredgewidth=MARKER_EDGEWIDTH,
    )


def _add_bw_legend(ax, include_cp, show_nocontention=False):
    tp_h, cp_h = _tp_cp_legend_handles()
    if include_cp:
        handles = [tp_h, cp_h]
        labels = ["TP", "CP"]
    else:
        handles = [tp_h]
        labels = ["TP"]
    if show_nocontention:
        handles.append(_tp_nocontention_legend_handle())
        labels.append("TP (no-cont.)")
    leg = ax.legend(
        handles,
        labels,
        fontsize=7,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#888888",
        facecolor="white",
        labelspacing=0.35,
    )
    leg.set_zorder(1)
    leg.get_frame().set_linewidth(0.8)


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


def _annotate_models(ax, model_groups, df, x_last, include_cp=True):
    ymax = ax.get_ylim()[1]
    dy = 0.01 * ymax
    x_text = x_last

    kinds = ("TP", "CP") if include_cp else ("TP",)

    for model, cols in model_groups:
        y_ends = []
        for kind in kinds:
            col = cols[kind]
            if col is None:
                continue
            y_ends.append(float(df[col].astype(float).values[-1]))
        if not y_ends:
            continue
        y_high = max(y_ends)
        y_top = ax.get_ylim()[1]
        y_text = min(y_high + dy, y_top * 0.995)
        ax.text(
            x_text,
            y_text,
            model,
            ha="right",
            va="bottom",
            fontsize=6,
            color="black",
            alpha=0.7,
            clip_on=True,
            zorder=4,
            fontweight="bold",
            fontstyle="italic",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot latency vs interconnect bandwidth (results/bw CSV)"
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
    parser.add_argument("--legend", action="store_true", help="Show TP/CP legend")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bw_dir = args.bw_dir if args.bw_dir else script_dir

    input_filename, out_filename = _resolve_bw_csv_paths(bw_dir, args)

    df = pd.read_csv(input_filename)
    df_nc_plot = None
    nc_ideal_row = None
    if _is_thor_device(args.device):
        nc_path = os.path.splitext(input_filename)[0] + "_nocontention.csv"
        if os.path.isfile(nc_path):
            df_nc = pd.read_csv(nc_path)
            nc_bw = df_nc.columns[0]
            df_nc_plot, nc_ideal_row = _split_data_and_ideal(df_nc, nc_bw)
            if df_nc_plot.empty:
                df_nc_plot = None
                nc_ideal_row = None
    if df.shape[1] < 2:
        raise ValueError(
            "CSV must have bandwidth column plus at least one series column"
        )

    bw_col = df.columns[0]
    df_plot, ideal_row = _split_data_and_ideal(df, bw_col)
    if df_plot.empty:
        raise ValueError("CSV has no numeric bandwidth rows to plot")
    x = df_plot[bw_col].astype(float).values
    if np.any(x <= 0):
        raise ValueError("Bandwidth values must be positive for log-scale x-axis")
    series_cols = list(df.columns[1:])
    model_groups, ungrouped_cols = _group_series_by_model(series_cols)
    include_cp = not _is_thor_device(args.device) and args.phase == "prefill"

    nc_ylim_cols = None
    if df_nc_plot is not None:
        nc_ylim_cols = [
            g["TP"]
            for _m, g in model_groups
            if g.get("TP") is not None and g["TP"] in df_nc_plot.columns
        ]
        if not nc_ylim_cols:
            df_nc_plot = None
            nc_ideal_row = None
            nc_ylim_cols = None

    plt.style.use("seaborn-v0_8-white")
    if args.phase == "prefill":
        fig, ax = plt.subplots(figsize=(1.7, 2.2))
    else:
        fig, ax = plt.subplots(figsize=(1.7, 1.9))

    _plot_bw_data_series(
        ax, x, df_plot, model_groups, ungrouped_cols, include_cp, df_nc_plot=df_nc_plot
    )

    ylim_cols = _series_cols_for_ylim(model_groups, ungrouped_cols, include_cp)
    _configure_bw_axes(
        ax,
        x,
        df_plot,
        ylim_cols,
        ideal_row,
        series_cols,
        include_cp,
        args.phase,
        df_nc_plot=df_nc_plot,
        nc_ideal_row=nc_ideal_row,
        nc_ylim_cols=nc_ylim_cols,
    )

    x_last = float(x[-1])
    _annotate_models(ax, model_groups, df_plot, x_last, include_cp=include_cp)

    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="--",
        color="#B8B8B8",
        linewidth=0.7,
        alpha=0.95,
        zorder=0,
    )
    ax.grid(
        axis="x",
        which="major",
        linestyle="--",
        color="#D8D8D8",
        linewidth=0.5,
        alpha=0.85,
        zorder=0,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)

    if args.legend:
        _add_bw_legend(
            ax,
            include_cp,
            show_nocontention=df_nc_plot is not None,
        )

    # plt.tight_layout()
    fig.subplots_adjust(left=0.34, right=0.93, top=0.93, bottom=0.27)
    plt.savefig(out_filename, dpi=300, facecolor="white")
    print(f"Chart successfully saved to: {out_filename}")


if __name__ == "__main__":
    main()
