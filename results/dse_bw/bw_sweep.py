#!/usr/bin/env python3
"""
Bandwidth sweep: total inference latency vs. single-link bandwidth (GB/s).

For a fixed (SM count, L2 size) hardware configuration, sweeps the UCIe
single-link bandwidth and plots total end-to-end latency for three workload
sizes:
  L  –  4-card  (Qwen3_8B,    2 links active, total BW = 2 × link BW)
  M  –  2-card  (Qwen3_4B,    1 link active,  total BW = 1 × link BW)
  S  –  1-card  (Qwen3_1_7B,  no link)

Bandwidth sweep ranges (single-link, GB/s):
  Orin : [4, 8, 16, 32]
  Thor : [16, 32, 64, 128]

Usage
-----
  python bw_sweep.py --base_hw Orin --inference_config Robo \\
      --sm_count 16 --l2_mb 4.0 --precision int8

  python bw_sweep.py --base_hw Thor --inference_config AD \\
      --sm_count 12 --l2_mb 32.0 --precision fp8 \\
      --mem_freq 8533 --mem_bitwidth 128
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    MultipleLocator,
    NullFormatter,
    NullLocator,
)

# ── Make the project root importable ────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import dse.dse as dse_module  # noqa: E402
from dse.dse import (  # noqa: E402, I001
    _DEFAULT_MEM,
    _LPDDR5_BITWIDTHS,
    _LPDDR5_FREQS,
    _LPDDR6_BITWIDTHS,
    _LPDDR6_FREQS,
    _MEM_BITWIDTH_CHOICES,
    _MEM_FREQ_CHOICES,
    LLM_PRECISION_CHOICES,
    _load_dse_cache,
    compute_inference_latency,
)

# ── Bandwidth sweep candidates (single-link, GB/s) ──────────────────────────
_BW_CANDIDATES_GBS = {
    "Orin": [4, 8, 16, 32],
    "Thor": [16, 32, 64, 128],
}

# ── Reference bandwidth for the ideal-line annotation (single-link, GB/s) ───
_REF_BW_GBS = 256

# ── Size → degree mapping ────────────────────────────────────────────────────
_SIZE_DEGREE = {"L": 4, "M": 2, "S": 1}

# ── Plot style (mirrors plot_perf.py conventions) ────────────────────────────
# Point colour encodes parallelism (TP/CP); line is always gray.
# Point shape encodes workload size (L / M).
_COLOR_TP = "#4F89BA"  # blue  – matches COLOR_TP in plot_perf.py
_COLOR_CP = "#D67F42"  # orange – matches COLOR_CP in plot_perf.py
_PARA_COLOR = {"TP": _COLOR_TP, "CP": _COLOR_CP}

_LINE_GRAY = "#B0B0B0"  # gray line for all sizes

_MARKER_SHAPE = {"L": "o", "M": "^"}  # circle for L, triangle for M
_MARKER_EDGEWIDTH = 0.85
_MARKER_SIZE = 5


# ── Argument parsing ──────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep single-link bandwidth (GB/s) and plot total inference "
            "latency for L/M/S workload sizes on a fixed hardware config."
        )
    )
    parser.add_argument(
        "--base_hw",
        required=True,
        choices=["Orin", "Thor"],
        help="Base hardware platform.",
    )
    parser.add_argument(
        "--inference_config",
        required=True,
        choices=["Robo", "AD"],
        help="Inference configuration (Robo: prefill-only; AD: with speculative decode).",
    )
    parser.add_argument(
        "--sm_count",
        type=int,
        required=True,
        help="Number of SMs in the GPU.",
    )
    parser.add_argument(
        "--l2_mb",
        type=float,
        required=True,
        help="L2 cache size in MB.",
    )
    parser.add_argument(
        "--precision",
        required=True,
        help=("LLM precision.  Orin: fp16_int4 | int8.  Thor: fp8 | fp4."),
    )
    parser.add_argument(
        "--mem_bitwidth",
        type=int,
        choices=_MEM_BITWIDTH_CHOICES,
        required=True,
        help=("Memory bus width in bits."),
    )
    parser.add_argument(
        "--mem_freq",
        type=int,
        choices=_MEM_FREQ_CHOICES,
        default=None,
        help=("Memory frequency in MT/s.  Default: 6400 for Orin, 8533 for Thor."),
    )
    args = parser.parse_args()

    # Apply hw-specific defaults for mem_freq / mem_bitwidth
    default_freq, default_bw = _DEFAULT_MEM[args.base_hw]
    args.mem_freq = args.mem_freq if args.mem_freq is not None else default_freq
    args.mem_bitwidth = (
        args.mem_bitwidth if args.mem_bitwidth is not None else default_bw
    )

    # Validate mem_freq / mem_bitwidth combination (mirrors dse.py logic)
    if args.mem_freq in _LPDDR5_FREQS and args.mem_bitwidth not in _LPDDR5_BITWIDTHS:
        parser.error(
            f"mem_freq={args.mem_freq} (LPDDR5/5x) requires "
            f"mem_bitwidth in {sorted(_LPDDR5_BITWIDTHS)}, got {args.mem_bitwidth}."
        )
    if args.mem_freq in _LPDDR6_FREQS and args.mem_bitwidth not in _LPDDR6_BITWIDTHS:
        parser.error(
            f"mem_freq={args.mem_freq} (LPDDR6) requires "
            f"mem_bitwidth in {sorted(_LPDDR6_BITWIDTHS)}, got {args.mem_bitwidth}."
        )

    # Validate LLM precision for the chosen platform (mirrors dse.py logic)
    valid_precisions = LLM_PRECISION_CHOICES[args.base_hw]
    if args.precision not in valid_precisions:
        parser.error(
            f"For {args.base_hw}, --precision must be one of {valid_precisions}. "
            f"Got: {args.precision!r}"
        )

    return args


# ── Simulation helpers ────────────────────────────────────────────────────────


def compute_ref_latencies(
    base_hw: str,
    inference_config: str,
    sm_count: int,
    l2_mb: float,
    mem_freq: int,
    mem_bitwidth: int,
    precision: str,
    dse_cache: dict,
    ref_bw_gbs: float = _REF_BW_GBS,
) -> dict:
    """Simulate L and M at *ref_bw_gbs* GB/s and return their total latencies.

    Returns
    -------
    dict  size → total_lat_ms  (keys "L" and "M")
    """
    link_bw_gbps = ref_bw_gbs * 8  # GB/s → Gbps
    dse_module._LINK_BW_GBPS = link_bw_gbps
    print(f"\n  Reference link BW = {ref_bw_gbs} GB/s  ({link_bw_gbps} Gbps)")

    ref_lats: dict = {}
    for size in ("L", "M"):
        r = compute_inference_latency(
            inference_config,
            size,
            base_hw,
            sm_count,
            l2_mb,
            mem_freq,
            mem_bitwidth,
            precision,
            dse_cache,
        )
        ref_lats[size] = r["total_lat_ms"]
        degree = _SIZE_DEGREE[size]
        print(
            f"    {inference_config}-{size} ({degree}-card): "
            f"{ref_lats[size]:.2f} ms  [ref]"
        )
    return ref_lats


def run_bw_sweep(
    base_hw: str,
    inference_config: str,
    sm_count: int,
    l2_mb: float,
    mem_freq: int,
    mem_bitwidth: int,
    precision: str,
    dse_cache: dict,
) -> tuple:
    """
    For each bandwidth point in _BW_CANDIDATES_GBS[base_hw], patch
    dse_module._LINK_BW_GBPS, simulate L and M workload sizes, and record
    total latency and LLM prefill parallelism (TP / CP).

    Returns
    -------
    latency_by_size : dict  size → list[float]   (total_lat_ms per BW point)
    para_by_size    : dict  size → list[str]      ("TP" or "CP" per BW point)

    Both dicts have keys "L" and "M" only (S is excluded from the plot).
    """
    bw_list_gbs = _BW_CANDIDATES_GBS[base_hw]
    latency_by_size = {"L": [], "M": []}
    para_by_size = {"L": [], "M": []}

    for bw_gbs in bw_list_gbs:
        link_bw_gbps = bw_gbs * 8  # GB/s → Gbps
        dse_module._LINK_BW_GBPS = link_bw_gbps
        print(f"\n  Link BW = {bw_gbs} GB/s  ({link_bw_gbps} Gbps)")

        for size in ("L", "M"):
            r = compute_inference_latency(
                inference_config,
                size,
                base_hw,
                sm_count,
                l2_mb,
                mem_freq,
                mem_bitwidth,
                precision,
                dse_cache,
            )
            lat_ms = r["total_lat_ms"]
            para = r["breakdown"]["llm_prefill_para"]
            latency_by_size[size].append(lat_ms)
            para_by_size[size].append(para)
            degree = _SIZE_DEGREE[size]
            print(
                f"    {inference_config}-{size} ({degree}-card): "
                f"{lat_ms:.2f} ms  [{para}]"
            )

    return latency_by_size, para_by_size


# ── Plot ──────────────────────────────────────────────────────────────────────


def _shape_legend_handle(size: str, inference_config: str) -> Line2D:
    """Legend entry encoding workload size via marker shape (gray line + marker)."""
    degree = _SIZE_DEGREE[size]
    return Line2D(
        [0],
        [0],
        color=_LINE_GRAY,
        linewidth=1.8,
        linestyle="-",
        marker=_MARKER_SHAPE[size],
        markersize=_MARKER_SIZE,
        markerfacecolor=_LINE_GRAY,
        markeredgecolor=_LINE_GRAY,
        markeredgewidth=_MARKER_EDGEWIDTH,
        label=f"{inference_config}-{size} ({degree}-Chip)",
    )


def _para_legend_handle(para: str) -> Line2D:
    """Legend entry encoding parallelism via marker colour (no line)."""
    color = _PARA_COLOR[para]
    return Line2D(
        [0],
        [0],
        color="none",
        linestyle="none",
        marker="o",
        markersize=_MARKER_SIZE,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=_MARKER_EDGEWIDTH,
        label=para,
    )


def plot_bw_sweep(
    bw_list_gbs: list,
    latency_by_size: dict,
    para_by_size: dict,
    inference_config: str,
    *,
    ref_lats: dict = None,
    out_path: str,
) -> None:
    """Plot total latency (ms) vs. single-link bandwidth (GB/s) for L and M.

    Gray lines connect points; marker colour encodes TP (blue) / CP (orange);
    marker shape encodes workload size (L: circle, M: triangle).
    If *ref_lats* is provided (size → ms), a gray dashed horizontal line is
    drawn for each size at the corresponding reference latency.
    """
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(2, 1.6))

    x = np.array(bw_list_gbs, dtype=float)

    for size in ("L", "M"):
        lats = latency_by_size[size]
        paras = para_by_size[size]

        # Gray connecting line (no markers)
        ax.plot(
            x,
            lats,
            color=_LINE_GRAY,
            linestyle="-",
            linewidth=1.8,
            zorder=1,
        )

        # Individual coloured markers (colour = TP/CP, shape = size)
        for xi, yi, para in zip(x, lats, paras):
            color = _PARA_COLOR[para]
            ax.plot(
                xi,
                yi,
                linestyle="none",
                marker=_MARKER_SHAPE[size],
                markersize=_MARKER_SIZE,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=_MARKER_EDGEWIDTH,
                zorder=3,
            )

    # ── X axis: log2 scale, ticks only at the candidate BW points ────────────
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:g}"
        )
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("D2D BW (GB/s)", fontsize=9, fontweight="bold")

    # ── Y axis ────────────────────────────────────────────────────────────────
    ax.set_ylabel("Total latency (ms)", fontsize=9, fontweight="bold")
    ax.set_xlim(x.min() / 2**0.25, x.max() * 2**0.25)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_ylim(70, 130)

    # ── Reference horizontal lines (one per size, gray dashed) ──────────────
    if ref_lats:
        for size in ("L", "M"):
            if size in ref_lats:
                ax.axhline(
                    y=ref_lats[size],
                    color="#888888",
                    linestyle="--",
                    linewidth=1.0,
                    zorder=1,
                    clip_on=True,
                )

    # ── Grid (x + y) ─────────────────────────────────────────────────────────
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

    # ── Legend 1 (upper-left): marker shape → workload size ──────────────────
    shape_handles = [
        _shape_legend_handle("L", inference_config),
        _shape_legend_handle("M", inference_config),
    ]
    leg1 = ax.legend(
        handles=shape_handles,
        labels=[h.get_label() for h in shape_handles],
        fontsize=7,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#888888",
        facecolor="white",
        labelspacing=0.35,
    )
    leg1.set_zorder(5)
    leg1.get_frame().set_linewidth(0.8)
    ax.add_artist(leg1)

    # ── Legend 2 (lower-right): marker colour → TP / CP ──────────────────────
    para_handles = [
        _para_legend_handle("TP"),
        _para_legend_handle("CP"),
    ]
    leg2 = ax.legend(
        handles=para_handles,
        labels=[h.get_label() for h in para_handles],
        fontsize=7,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.1),
        frameon=True,
        fancybox=False,
        edgecolor="#888888",
        facecolor="white",
        labelspacing=0.35,
    )
    leg2.set_zorder(5)
    leg2.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.28, right=0.95, top=0.95, bottom=0.22)
    plt.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()

    mem_bw_gbs = args.mem_freq * args.mem_bitwidth / 8 / 1000
    print("=" * 72)
    print("  Bandwidth Sweep")
    print("=" * 72)
    print(f"  Base HW          : {args.base_hw}")
    print(f"  Inference config : {args.inference_config}")
    print(f"  SM count         : {args.sm_count}")
    print(f"  L2 size          : {args.l2_mb} MB")
    print(f"  Precision        : {args.precision}")
    print(
        f"  Memory           : {args.mem_freq} MT/s × {args.mem_bitwidth}-bit"
        f"  ({mem_bw_gbs:.1f} GB/s)"
    )
    print(
        f"  Link BW range    : {_BW_CANDIDATES_GBS[args.base_hw]} GB/s  (single link)"
    )
    print("-" * 72)

    dse_cache = _load_dse_cache()

    print(f"\n  Computing reference latencies at {_REF_BW_GBS} GB/s ...")
    ref_lats = compute_ref_latencies(
        base_hw=args.base_hw,
        inference_config=args.inference_config,
        sm_count=args.sm_count,
        l2_mb=args.l2_mb,
        mem_freq=args.mem_freq,
        mem_bitwidth=args.mem_bitwidth,
        precision=args.precision,
        dse_cache=dse_cache,
    )

    latency_by_size, para_by_size = run_bw_sweep(
        base_hw=args.base_hw,
        inference_config=args.inference_config,
        sm_count=args.sm_count,
        l2_mb=args.l2_mb,
        mem_freq=args.mem_freq,
        mem_bitwidth=args.mem_bitwidth,
        precision=args.precision,
        dse_cache=dse_cache,
    )

    tag = (
        f"{args.base_hw}_{args.inference_config}_{args.precision}_"
        f"sm{args.sm_count}_l2_{args.l2_mb}mb_"
        f"{args.mem_freq}MHz_{args.mem_bitwidth}bit"
    ).replace("/", "_")
    out_path = os.path.join(_THIS_DIR, f"bw_sweep_{tag}.png")

    print("\n  Generating bandwidth sweep line chart ...")
    plot_bw_sweep(
        _BW_CANDIDATES_GBS[args.base_hw],
        latency_by_size,
        para_by_size,
        args.inference_config,
        ref_lats=ref_lats,
        out_path=out_path,
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
