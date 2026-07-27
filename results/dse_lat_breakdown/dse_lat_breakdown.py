#!/usr/bin/env python3
"""
DSE latency breakdown bar chart.

Takes ONE hardware configuration directly via --sm / --l2 (no grid search,
no area-based auto-selection) and plots a stacked bar chart showing the
per-size latency breakdown for that configuration.

Three bars (L / M / S), each stacked as:
  ViT   (bottom)
  LLM prefill
  LLM decode  (only present for the AD inference config)

A red dashed line at 100 ms is drawn and labelled "10 FPS".

Output
------
  breakdown_<tag>.png   saved in the same directory as this script.

Usage
-----
  python dse_lat_breakdown.py --sm 32 --l2 8 --base_hw Orin \\
      --inference_config Robo --precision fp16_int4

  python dse_lat_breakdown.py --sm 24 --l2 32 --base_hw Thor \\
      --inference_config AD --precision fp8 \\
      --mem_freq 10667 --mem_bitwidth 128 --latency_limit 120
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ── Make the project root importable ────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dse.dse import (  # noqa: E402, I001
    compute_inference_latency,
    _load_dse_cache,
    _check_area_constraint,
    GPU_AREA_FRACTION,
    LLM_PRECISION_CHOICES,
    _MEM_FREQ_CHOICES,
    _MEM_BITWIDTH_CHOICES,
    _DEFAULT_MEM,
    _LPDDR5_FREQS,
    _LPDDR6_FREQS,
    _LPDDR5_BITWIDTHS,
    _LPDDR6_BITWIDTHS,
)

# ── Plot constants ───────────────────────────────────────────────────────────

_FPS10_MS = 100.0  # 10 FPS ↔ 100 ms

_SEG_COLORS = {
    "vit": "#A3C8EF",  # TP light blue
    "prefill": "#4F89BA",  # TP dark blue
    "decode": "#D67F42",  # CP dark orange
}
_SEG_LABELS = {
    "vit": "ViT",
    "prefill": "LLM Prefill",
    "decode": "LLM Decode",
}

_SIZE_DEGREE = {"L": 4, "M": 2, "S": 1}


# ── Plot ─────────────────────────────────────────────────────────────────────


def plot_breakdown_bar(
    breakdown_by_size: dict,
    inference_config: str,
    sm: int,
    l2: float,
    gpu_area: float,
    latency_limit: float,
    *,
    total_lat_no_overlap_by_size: dict = None,
    out_path: str,
):
    """Draw a stacked bar chart of latency breakdown for the chosen config.

    breakdown_by_size: {"L": {"vit": ms, "prefill": ms, "decode": ms}, ...}
    (this is the --overlap result set)

    total_lat_no_overlap_by_size, if given: {"L": total_ms, ...} — the total
    end-to-end latency *without* --overlap (fully serialized comm) for the
    same hardware config.  Drawn as a hollow bar (dark-gray outline only,
    no fill) overlaid on top of the same x position, so the extra height
    above the stacked bar shows the latency exposed by disabling overlap.
    """
    plt.style.use("seaborn-v0_8-white")

    sizes = ["S", "M", "L"]
    has_decode = any(breakdown_by_size[s]["decode"] > 0 for s in sizes)

    segments = ["vit", "prefill"] + (["decode"] if has_decode else [])

    fig, ax = plt.subplots(figsize=(2.5, 2))

    x = np.arange(len(sizes)) * 0.5
    bar_w = 0.25
    bottoms = np.zeros(len(sizes))
    margin = bar_w * 1.5

    for seg in segments:
        vals = np.array([breakdown_by_size[s][seg] for s in sizes])
        bars = ax.bar(
            x,
            vals,
            bar_w,
            bottom=bottoms,
            color=_SEG_COLORS[seg],
            label=_SEG_LABELS[seg],
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        # Annotate each segment with its value (skip tiny slivers)
        for bar, val, bot in zip(bars, vals, bottoms):
            if val < 2.0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bot + val / 2,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
        bottoms += vals

    # Hollow, dark-gray-outline bars for the no-overlap total latency,
    # overlaid at the same x position/width as the stacked bars above.
    no_overlap_vals = None
    if total_lat_no_overlap_by_size is not None:
        no_overlap_vals = np.array([total_lat_no_overlap_by_size[s] for s in sizes])
        ax.bar(
            x,
            no_overlap_vals,
            bar_w,
            facecolor="none",
            edgecolor="#404040",
            linewidth=1.3,
            zorder=4,
        )

    # Figure out the y-axis range now so label offsets (below) can be sized
    # proportionally and never collide, regardless of the latency scale.
    max_val = bottoms.max()
    if no_overlap_vals is not None:
        max_val = max(max_val, no_overlap_vals.max())
    y_top = max(180.0, max_val + 20)
    base_offset = y_top * 0.006  # small gap above a bar's own top edge
    min_label_gap = y_top * 0.09  # min vertical gap between the two labels

    # Annotate total latency (--overlap) above each bar
    for xi, total_ms in zip(x, bottoms):
        ax.text(
            xi,
            total_ms + base_offset,
            f"{total_ms:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            zorder=6,
        )

    # No-overlap total label: keep it clear of the filled-bar label directly
    # below it when the two totals are close together.  Skip it entirely
    # when the no-overlap total equals the --overlap total (e.g. a 1-chip
    # config has no inter-chip comm to overlap in the first place), since
    # the two labels would just be duplicates.
    if no_overlap_vals is not None:
        for xi, total_ms, filled_total in zip(x, no_overlap_vals, bottoms):
            if abs(total_ms - filled_total) < 0.05:
                continue
            label_y = max(total_ms + base_offset, filled_total + min_label_gap)
            ax.text(
                xi,
                label_y,
                f"{total_ms:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#404040",
                fontweight="bold",
                zorder=6,
            )

    # 10 FPS reference line
    ax.axhline(
        _FPS10_MS,
        color="#B22222",
        linestyle="--",
        linewidth=0.9,
        zorder=1,
        alpha=0.95,
    )
    txt = ax.text(
        x[0] - margin + 0.02,
        _FPS10_MS - 22,
        "10\nFPS",
        color="#B22222",
        fontsize=6,
        fontstyle="italic",
        ha="left",
        va="bottom",
        zorder=1,
    )
    txt.set_linespacing(0.9)

    degree_labels = {s: _SIZE_DEGREE[s] for s in sizes}
    x_labels = [f"{inference_config}-{s}\n({degree_labels[s]}-Chip)" for s in sizes]
    ax.set_xlim(x[0] - margin, x[-1] + margin)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)

    ax.set_ylabel("Latency (ms)", fontsize=9, fontweight="bold")
    ax.set_ylim(bottom=0, top=y_top)
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

    handles, labels = ax.get_legend_handles_labels()
    if no_overlap_vals is not None:
        handles.append(Patch(facecolor="none", edgecolor="#404040", linewidth=1.3))
        labels.append("Serialized total")
    leg = ax.legend(
        handles,
        labels,
        fontsize=7,
        loc="upper left",
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#888888",
        facecolor="white",
        labelspacing=0.2,
        columnspacing=0.8,
        handletextpad=0.4,
        handlelength=1.2,
    )
    leg.set_zorder(5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the latency breakdown bar chart for ONE hardware "
            "configuration, specified directly via --sm / --l2 "
            "(no grid search, no area-based auto-selection)."
        )
    )
    parser.add_argument("--sm", type=int, required=True, help="SM count.")
    parser.add_argument(
        "--l2", type=float, required=True, help="L2 cache size in MB."
    )
    parser.add_argument(
        "--base_hw",
        required=True,
        choices=["Orin", "Thor"],
        help="Base hardware platform whose config JSON is used as the starting point.",
    )
    parser.add_argument(
        "--inference_config",
        required=True,
        choices=["Robo", "AD"],
        help="Robo: prefill-only pipeline.  AD: pipeline with speculative decode.",
    )
    parser.add_argument(
        "--precision",
        required=True,
        help=(
            "LLM precision.  "
            "Orin: fp16_int4 (fp16 prefill / int4 decode) | int8.  "
            "Thor: fp8 | fp4."
        ),
    )
    parser.add_argument(
        "--latency_limit",
        type=float,
        default=100.0,
        help="End-to-end latency constraint in ms, for reference only (default: 100 ms).",
    )
    parser.add_argument(
        "--overlap",
        action="store_true",
        help="Model ideal compute/comm overlap: exposed comm latency = "
        "max(0, T_comm - T_comp). Default is no overlap (fully serialized).",
    )
    parser.add_argument(
        "--mem_freq",
        type=int,
        choices=_MEM_FREQ_CHOICES,
        default=None,
        help=(
            "Memory frequency in MT/s.  "
            "LPDDR5/5x: 6400 | 8533 | 9600 (bitwidth must be 128 or 256).  "
            "LPDDR6:    10667 | 12800       (bitwidth must be 128 or 256).  "
            "Default: 6400 for Orin, 8533 for Thor."
        ),
    )
    parser.add_argument(
        "--mem_bitwidth",
        type=int,
        choices=_MEM_BITWIDTH_CHOICES,
        default=None,
        help=(
            "Memory bus width in bits.  "
            "LPDDR5/5x: 128 | 256.  LPDDR6: 128 | 256.  "
            "Default: 256 for both Orin and Thor."
        ),
    )
    args = parser.parse_args()

    # ── Apply defaults for mem_freq / mem_bitwidth based on base_hw ──────────
    default_freq, default_bw = _DEFAULT_MEM[args.base_hw]
    args.mem_freq = args.mem_freq if args.mem_freq is not None else default_freq
    args.mem_bitwidth = (
        args.mem_bitwidth if args.mem_bitwidth is not None else default_bw
    )

    # ── Validate mem_freq / mem_bitwidth combination ─────────────────────────
    if args.mem_freq in _LPDDR5_FREQS and args.mem_bitwidth not in _LPDDR5_BITWIDTHS:
        parser.error(
            f"mem_freq={args.mem_freq} (LPDDR5/5x) is only compatible with "
            f"mem_bitwidth in {sorted(_LPDDR5_BITWIDTHS)}, got {args.mem_bitwidth}."
        )
    if args.mem_freq in _LPDDR6_FREQS and args.mem_bitwidth not in _LPDDR6_BITWIDTHS:
        parser.error(
            f"mem_freq={args.mem_freq} (LPDDR6) is only compatible with "
            f"mem_bitwidth in {sorted(_LPDDR6_BITWIDTHS)}, got {args.mem_bitwidth}."
        )

    # ── Validate LLM precision for the chosen platform ───────────────────────
    valid_precisions = LLM_PRECISION_CHOICES[args.base_hw]
    if args.precision not in valid_precisions:
        parser.error(
            f"For {args.base_hw}, --precision must be one of {valid_precisions}. "
            f"Got: {args.precision!r}"
        )

    return args


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    args = _parse_args()
    sm, l2 = args.sm, args.l2
    inference_config = args.inference_config
    latency_limit = args.latency_limit

    # soc_area_mm2=inf → skip the area-budget check, just get the GPU die area.
    _, gpu_area = _check_area_constraint(args.base_hw, sm, l2, float("inf"))
    # Minimum SoC area required to keep this GPU within the GPU_AREA_FRACTION
    # budget (GPU ≤ GPU_AREA_FRACTION × SoC area).
    soc_area = gpu_area / GPU_AREA_FRACTION
    print(
        f"\n  Config: SM={sm}  L2={l2:.1f} MB  |  "
        f"GPU area={gpu_area:.1f} mm²  |  "
        f"required SoC area≥{soc_area:.1f} mm² "
        f"(GPU_AREA_FRACTION={GPU_AREA_FRACTION:.0%})"
    )

    dse_cache = _load_dse_cache()

    # Collect latency breakdown for each size, plus the total end-to-end
    # latency *without* --overlap (fully serialized comm) for the same
    # hardware config, for the hollow-outline overlay bars.
    print("\n  Computing latency breakdown for L / M / S …")
    breakdown_by_size = {}
    total_lat_no_overlap_by_size = {}
    for size in ("L", "M", "S"):
        result = compute_inference_latency(
            inference_config,
            size,
            args.base_hw,
            sm,
            l2,
            args.mem_freq,
            args.mem_bitwidth,
            args.precision,
            dse_cache,
            overlap=args.overlap,
        )
        breakdown = result["breakdown"]
        breakdown_by_size[size] = {
            "vit": breakdown.get("vit_lat_ms", 0.0),
            "prefill": breakdown.get("llm_prefill_lat_ms", 0.0),
            "decode": breakdown.get("llm_decode_lat_ms", 0.0),
        }

        no_overlap_result = compute_inference_latency(
            inference_config,
            size,
            args.base_hw,
            sm,
            l2,
            args.mem_freq,
            args.mem_bitwidth,
            args.precision,
            dse_cache,
            overlap=False,
        )
        total_lat_no_overlap_by_size[size] = no_overlap_result["total_lat_ms"]

    tag = (
        f"{args.base_hw}_{inference_config}_{args.precision}_"
        f"sm{sm}_l2_{l2:.1f}mb_{args.mem_freq}MHz_"
        f"{args.mem_bitwidth}bit_{latency_limit:.0f}ms"
    ).replace("/", "_")

    out_path = os.path.join(_THIS_DIR, f"breakdown_{tag}.png")
    print("\n  Generating breakdown bar chart …")
    plot_breakdown_bar(
        breakdown_by_size,
        inference_config,
        sm,
        l2,
        gpu_area,
        latency_limit,
        total_lat_no_overlap_by_size=total_lat_no_overlap_by_size,
        out_path=out_path,
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
