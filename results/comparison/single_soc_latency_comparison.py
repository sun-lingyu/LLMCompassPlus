#!/usr/bin/env python3
"""
Single-SoC (no chip parallelism) latency comparison bar chart.

Given ONE base hardware platform (Orin / Thor), compares TWO full hardware
configurations — (SM count, L2 size, memory frequency, memory bitwidth) —
by running the AD-S / AD-M / AD-L inference pipelines entirely on a SINGLE
SoC (chip parallelism degree forced to 1 — i.e. the LLM is never split
across multiple chips, even for AD-L which normally uses a 4-card
tensor-parallel degree).

The chart has three groups (AD-S / AD-M / AD-L), each with two bars — one per
(SM, L2) configuration — showing ONLY the total end-to-end latency (no
per-segment breakdown). A red dashed line is drawn at 100 ms.

The GPU die area for both configurations is printed to stdout.

Output
------
  single_soc_lat_<tag>.png   saved in the same directory as this script.

Usage
-----
  python single_soc_latency_comparison.py --base_hw Thor \\
      --sm1 24 --l2_1 32 --mem_freq1 8533 --mem_bitwidth1 256 \\
      --sm2 40 --l2_2 16 --mem_freq2 10667 --mem_bitwidth2 128 \\
      --inference_config AD --precision fp8

  python single_soc_latency_comparison.py --base_hw Orin \\
      --sm1 16 --l2_1 4 --sm2 32 --l2_2 8 \\
      --inference_config AD --precision fp16_int4
      # mem_freq*/mem_bitwidth* default to the platform's default memory
      # spec (same for both configs) when omitted.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# ── Make the project root importable ────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dse import dse as _dse  # noqa: E402
from dse.dse import (  # noqa: E402, I001
    _DEFAULT_MEM,
    _LPDDR5_BITWIDTHS,
    _LPDDR5_FREQS,
    _LPDDR6_BITWIDTHS,
    _LPDDR6_FREQS,
    _MEM_BITWIDTH_CHOICES,
    _MEM_FREQ_CHOICES,
    LLM_PRECISION_CHOICES,
    _check_area_constraint,
    _load_dse_cache,
    compute_inference_latency,
)

# ── Plot constants ───────────────────────────────────────────────────────────

_FPS10_MS = 100.0  # red reference line at 100 ms

# Two colors for the two (SM, L2) configurations being compared.
_CFG_COLORS = ["#4F89BA", "#D67F42"]

_SIZES = ["S", "M", "L"]


def _mem_bw_gbps(mem_freq: int, mem_bitwidth: int) -> float:
    """Peak memory bandwidth in GB/s given MT/s and bus width in bits."""
    return mem_freq * mem_bitwidth / 8 / 1000


# ── Single-SoC (degree forced to 1) latency computation ────────────────────


def compute_single_soc_latency(
    inference_config: str,
    size: str,
    base_hw: str,
    sm_count: int,
    l2_mb: float,
    mem_freq: int,
    mem_bitwidth: int,
    llm_precision: str,
    dse_cache: dict,
) -> dict:
    """
    Same as dse.compute_inference_latency, but forces the LLM tensor-parallel
    degree to 1 (single SoC, no chip parallelism) regardless of the size's
    normal degree (AD-L is normally 4-card, AD-M 2-card).

    Temporarily patches the module-level _INFERENCE_CONFIGS entry's "degree"
    field (restored in a finally block), since degree is baked into the
    hardcoded per-size config used internally by compute_inference_latency.
    Degree is also part of the DSE cache key, so degree=1 results are cached
    separately from the normal multi-chip results and never collide.
    """
    cfg = _dse._INFERENCE_CONFIGS[inference_config][size]
    original_degree = cfg["degree"]
    cfg["degree"] = 1
    try:
        return compute_inference_latency(
            inference_config,
            size,
            base_hw,
            sm_count,
            l2_mb,
            mem_freq,
            mem_bitwidth,
            llm_precision,
            dse_cache,
            overlap=False,  # irrelevant at degree=1: no inter-chip comm at all
        )
    finally:
        cfg["degree"] = original_degree


# ── Plot ─────────────────────────────────────────────────────────────────────


def plot_comparison_bar(
    lat_by_cfg: list,
    cfg_labels: list,
    inference_config: str,
    out_path: str,
):
    """Draw a grouped bar chart comparing two HW configs across S/M/L sizes.

    lat_by_cfg: [{"S": ms, "M": ms, "L": ms}, {"S": ms, "M": ms, "L": ms}]
    cfg_labels: ["SM=..\\nL2=..MB\\nMem=..GB/s", ...]  (one per config, same order)
    """
    plt.style.use("seaborn-v0_8-white")

    n_cfg = len(lat_by_cfg)
    group_w = 0.5
    bar_w = group_w / n_cfg * 0.85
    offsets = np.linspace(-group_w / 2 + bar_w / 2, group_w / 2 - bar_w / 2, n_cfg)

    fig, ax = plt.subplots(figsize=(3.5, 1.3))

    x_group = np.arange(len(_SIZES)) * 0.9
    margin = group_w * 1.1

    # Figure out the y-axis range up front (before drawing bars/labels) so
    # the legend box in the top-left corner never collides with the tallest
    # bar's value label, regardless of the latency scale.
    max_val = max(max(vals_by_size.values()) for vals_by_size in lat_by_cfg)
    y_top = max(180.0, max_val * 1.4)
    base_offset = y_top * 0.012  # small gap above a bar's own top edge

    all_vals = []
    for i, (vals_by_size, label) in enumerate(zip(lat_by_cfg, cfg_labels)):
        vals = np.array([vals_by_size[s] for s in _SIZES])
        all_vals.append(vals)
        xi = x_group + offsets[i]
        bars = ax.bar(
            xi,
            vals,
            bar_w,
            color=_CFG_COLORS[i % len(_CFG_COLORS)],
            label=label,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + base_offset,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
                zorder=6,
            )

    # Red reference line at 100 ms
    ax.axhline(
        _FPS10_MS,
        color="#B22222",
        linestyle="--",
        linewidth=0.9,
        zorder=1,
        alpha=0.95,
    )
    # txt = ax.text(
    #     x_group[0] - margin + 0.02,
    #     _FPS10_MS + y_top * 0.015,
    #     "100 ms",
    #     color="#B22222",
    #     fontsize=10,
    #     fontstyle="italic",
    #     ha="left",
    #     va="bottom",
    #     zorder=1,
    # )
    # txt.set_linespacing(0.9)

    x_labels = [f"{inference_config}-{s}" for s in _SIZES]
    ax.set_xlim(x_group[0] - margin, x_group[-1] + margin)
    ax.set_xticks(x_group)
    ax.set_xticklabels(x_labels, fontsize=10, fontweight="bold")

    ax.set_ylabel("Latency (ms)", fontsize=10, fontweight="bold")
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

    leg = ax.legend(
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        ncol=1,
        frameon=False,
        labelspacing=1.0,
        handletextpad=0.5,
        handlelength=1.2,
    )
    leg.set_zorder(5)

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        bbox_extra_artists=(leg,),
    )
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two (SM, L2) hardware configurations on a SINGLE SoC "
            "(no chip parallelism) across AD-S/AD-M/AD-L total latency."
        )
    )
    parser.add_argument(
        "--base_hw",
        required=True,
        choices=["Orin", "Thor"],
        help="Base hardware platform whose config JSON is used as the starting point.",
    )
    parser.add_argument("--sm1", type=int, required=True, help="SM count, config 1.")
    parser.add_argument(
        "--l2_1", type=float, required=True, help="L2 cache size in MB, config 1."
    )
    parser.add_argument(
        "--mem_freq1",
        type=int,
        choices=_MEM_FREQ_CHOICES,
        default=None,
        help=(
            "Memory frequency in MT/s, config 1.  "
            "LPDDR5/5x: 6400 | 8533 | 9600 (bitwidth must be 128 or 256).  "
            "LPDDR6:    10667 | 12800       (bitwidth must be 128 or 256).  "
            "Default: 6400 for Orin, 8533 for Thor."
        ),
    )
    parser.add_argument(
        "--mem_bitwidth1",
        type=int,
        choices=_MEM_BITWIDTH_CHOICES,
        default=None,
        help=(
            "Memory bus width in bits, config 1.  "
            "LPDDR5/5x: 128 | 256.  LPDDR6: 128 | 256.  "
            "Default: 256 for both Orin and Thor."
        ),
    )
    parser.add_argument("--sm2", type=int, required=True, help="SM count, config 2.")
    parser.add_argument(
        "--l2_2", type=float, required=True, help="L2 cache size in MB, config 2."
    )
    parser.add_argument(
        "--mem_freq2",
        type=int,
        choices=_MEM_FREQ_CHOICES,
        default=None,
        help="Memory frequency in MT/s, config 2.  Same choices/rules as --mem_freq1.",
    )
    parser.add_argument(
        "--mem_bitwidth2",
        type=int,
        choices=_MEM_BITWIDTH_CHOICES,
        default=None,
        help="Memory bus width in bits, config 2.  Same choices/rules as --mem_bitwidth1.",
    )
    parser.add_argument(
        "--inference_config",
        default="AD",
        choices=["Robo", "AD"],
        help="Robo: prefill-only pipeline.  AD: pipeline with speculative decode "
        "(default: AD).",
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
    args = parser.parse_args()

    # ── Apply defaults for mem_freq / mem_bitwidth (per config) based on base_hw ──
    default_freq, default_bw = _DEFAULT_MEM[args.base_hw]
    args.mem_freq1 = args.mem_freq1 if args.mem_freq1 is not None else default_freq
    args.mem_bitwidth1 = (
        args.mem_bitwidth1 if args.mem_bitwidth1 is not None else default_bw
    )
    args.mem_freq2 = args.mem_freq2 if args.mem_freq2 is not None else default_freq
    args.mem_bitwidth2 = (
        args.mem_bitwidth2 if args.mem_bitwidth2 is not None else default_bw
    )

    # ── Validate mem_freq / mem_bitwidth combination, for each config ────────
    for freq, bw, tag in (
        (args.mem_freq1, args.mem_bitwidth1, "1"),
        (args.mem_freq2, args.mem_bitwidth2, "2"),
    ):
        if freq in _LPDDR5_FREQS and bw not in _LPDDR5_BITWIDTHS:
            parser.error(
                f"mem_freq{tag}={freq} (LPDDR5/5x) is only compatible with "
                f"mem_bitwidth{tag} in {sorted(_LPDDR5_BITWIDTHS)}, got {bw}."
            )
        if freq in _LPDDR6_FREQS and bw not in _LPDDR6_BITWIDTHS:
            parser.error(
                f"mem_freq{tag}={freq} (LPDDR6) is only compatible with "
                f"mem_bitwidth{tag} in {sorted(_LPDDR6_BITWIDTHS)}, got {bw}."
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
    inference_config = args.inference_config
    configs = [
        (args.sm1, args.l2_1, args.mem_freq1, args.mem_bitwidth1),
        (args.sm2, args.l2_2, args.mem_freq2, args.mem_bitwidth2),
    ]

    dse_cache = _load_dse_cache()

    print(f"\n{'=' * 72}")
    print(
        f"  Single-SoC (no parallelism) latency comparison  —  "
        f"{args.base_hw} / {inference_config} / {args.precision}"
    )
    print(f"{'=' * 72}")

    gpu_areas = []
    cfg_labels = []
    for sm, l2, mem_freq, mem_bitwidth in configs:
        # soc_area_mm2=inf → skip the area-budget check, just get the GPU die area.
        # (GPU area depends only on SM/L2, not on memory freq/bitwidth.)
        _, gpu_area = _check_area_constraint(args.base_hw, sm, l2, float("inf"))
        gpu_areas.append(gpu_area)
        mem_bw = _mem_bw_gbps(mem_freq, mem_bitwidth)
        cfg_labels.append(f"SM={sm}\nL2={l2:.1f}MB\nMem={mem_bw:.0f}GB/s")
        print(
            f"  Config: SM={sm:<4d} L2={l2:>5.1f} MB  "
            f"mem={mem_freq}MT/s x{mem_bitwidth}bit ({mem_bw:.1f} GB/s)  |  "
            f"GPU area={gpu_area:.1f} mm²"
        )
    print(f"{'=' * 72}\n")

    print("  Computing single-SoC (degree=1) latency for AD-S / AD-M / AD-L …")
    lat_by_cfg = []
    for sm, l2, mem_freq, mem_bitwidth in configs:
        lat_by_size = {}
        for size in _SIZES:
            result = compute_single_soc_latency(
                inference_config,
                size,
                args.base_hw,
                sm,
                l2,
                mem_freq,
                mem_bitwidth,
                args.precision,
                dse_cache,
            )
            lat_by_size[size] = result["total_lat_ms"]
            print(f"  {inference_config}-{size}={result['total_lat_ms']:.1f}ms")
        lat_by_cfg.append(lat_by_size)
        print(
            f"  SM={sm:<4d} L2={l2:>5.1f} MB  mem={mem_freq}MT/s x{mem_bitwidth}bit  |  "
            + "  ".join(
                f"{inference_config}-{s}={lat_by_size[s]:.1f}ms" for s in _SIZES
            )
        )

    tag = (
        f"{args.base_hw}_{inference_config}_{args.precision}_"
        f"sm{args.sm1}_l2{args.l2_1:.1f}mb_{args.mem_freq1}MHz_{args.mem_bitwidth1}bit_vs_"
        f"sm{args.sm2}_l2{args.l2_2:.1f}mb_{args.mem_freq2}MHz_{args.mem_bitwidth2}bit"
    ).replace("/", "_")

    out_path = os.path.join(_THIS_DIR, f"single_soc_lat_{tag}.png")
    print("\n  Generating comparison bar chart …")
    plot_comparison_bar(lat_by_cfg, cfg_labels, inference_config, out_path=out_path)

    print("\n  Done.")


if __name__ == "__main__":
    main()
