#!/usr/bin/env python3
"""
DSE 2D combined heatmap visualizer (2-in-1).

Runs the SM × L2 grid search via run_dse() and plots a single heatmap:
  - Cell color  → Latency (ms), fixed scale 20–100 ms – Blues (no colorbar on heatmap)
  - Upper label → Latency value (no unit)
  - Lower label → Power value with unit (W)

Configs that fail the area constraint or latency limit are shown in gray.
Only latency-passing configs receive a color.

Usage (examples)
----------------
  python dse_visualize_2in1.py --area 400 --base_hw Orin \
      --inference_config Robo-S --precision fp16_int4

  python dse_visualize_2in1.py --area 200 --base_hw Thor \
      --inference_config AD-S --precision fp8 \
      --mem_freq 10667 --mem_bitwidth 128 \
      --llm_model Qwen3_4B --latency_limit 120
"""

import argparse
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase

# ── Make the project root importable ────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dse.dse import (  # noqa: E402, I001
    run_dse,
    ORIN_SM_CANDIDATES,
    ORIN_L2_MB_CANDIDATES,
    THOR_SM_CANDIDATES,
    THOR_L2_MB_CANDIDATES,
    _MODEL_DEFAULT_DEGREE,
    _DEFAULT_MEM,
    LLM_PRECISION_CHOICES,
    _LPDDR5_FREQS,
    _LPDDR6_FREQS,
    _LPDDR5_BITWIDTHS,
    _LPDDR6_BITWIDTHS,
)


# ── Plotting helpers ─────────────────────────────────────────────────────────

# Latency colormap range (ms): heatmap + standalone colorbar share this scale.
LAT_CMAP_MS_MIN = 20.0
LAT_CMAP_MS_MAX = 100.0

CHIP_NUM = 4


def _make_gray_cmap(base_cmap: str):
    """
    Return a colormap that uses *base_cmap* for valid data and a light gray
    for NaN values (configs that fail area or latency constraints).
    """
    cmap = plt.get_cmap(base_cmap).copy()
    cmap.set_bad(color="#cccccc")
    return cmap


def _annotate_cells_combined(ax, lat_masked, pwr_masked, lat_vmin, lat_vmax):
    """Write two lines in every non-masked cell.

    Upper line: latency value (no unit), e.g. "42.3"
    Lower line: power value with unit (W), e.g. "18.7W"
    Text color adapts to latency cell brightness (blue colormap).
    """
    nrows, ncols = lat_masked.shape
    for i in range(nrows):
        for j in range(ncols):
            lat_val = lat_masked[i, j]
            pwr_val = pwr_masked[i, j]
            if np.ma.is_masked(lat_val) or np.isnan(lat_val):
                continue
            if lat_vmax > lat_vmin:
                normalized = (float(lat_val) - lat_vmin) / (lat_vmax - lat_vmin)
                normalized = max(0.0, min(1.0, normalized))
            else:
                normalized = 0.5
            text_color = "white" if normalized > 0.6 else "black"
            ax.text(
                j,
                i - 0.15,
                f"{float(lat_val):.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )
            ax.text(
                j,
                i + 0.25,
                f"{float(pwr_val):.1f}W",
                ha="center",
                va="center",
                fontsize=6,
                fontstyle="italic",
                color=text_color,
            )


def plot_combined_heatmap(
    lat_masked,
    pwr_masked,
    sm_list,
    l2_list,
    *,
    out_path: str,
):
    """Draw a single 2D heatmap colored by latency, annotated with latency + power.

    Latency color scale is fixed to LAT_CMAP_MS_MIN..LAT_CMAP_MS_MAX (ms); no colorbar.

    Each cell shows:
      upper line → latency value (no unit)
      lower line → power value with unit (W)

    pwr_masked / lat_masked shape: (nL2, nSM)
    sm_list: SM candidates (X-axis, ascending left→right)
    l2_list: L2 candidates (Y-axis, descending top→bottom)
    """
    plt.style.use("seaborn-v0_8-white")

    # sm_list is empty when no config passes the latency constraint.
    no_feasible = len(sm_list) == 0

    nrows = len(l2_list)
    # lat_masked always has at least 1 column (see build_heatmap_arrays).
    ncols = lat_masked.shape[1]

    cell_w = 0.35  # inches per SM column
    cell_h = 0.25  # inches per L2 row (1.5× original 0.2 for two-line annotation)
    fig_w = ncols * cell_w + 0.6
    fig_h = nrows * cell_h + 0.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap_obj = _make_gray_cmap("Blues")
    ax.imshow(
        lat_masked,
        cmap=cmap_obj,
        vmin=LAT_CMAP_MS_MIN,
        vmax=LAT_CMAP_MS_MAX,
        aspect="auto",
        origin="upper",
    )

    _annotate_cells_combined(
        ax, lat_masked, pwr_masked, LAT_CMAP_MS_MIN, LAT_CMAP_MS_MAX
    )

    ax.set_xlabel("SM Count", fontsize=7, fontweight="bold")
    if no_feasible:
        # Use an invisible placeholder label (same fontsize) so tight_layout
        # reserves the same vertical space as a normal x-axis tick row.
        ax.set_xticks([0])
        ax.set_xticklabels(["0"], fontsize=7, color="white")
    else:
        ax.set_xticks(np.arange(ncols))
        ax.set_xticklabels([str(s) for s in sm_list], fontsize=7)

    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels([f"{int(v)}" for v in l2_list], fontsize=7)
    ax.set_ylabel("L2 Size (MB)", fontsize=7, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=1.5)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_standalone_latency_colorbar(out_path: str) -> None:
    """Save a vertical Blues colorbar for latency (ms), same scale as the heatmap."""
    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(0.35, 3.0))
    ax_cbar = fig.add_axes([0.22, 0.12, 0.38, 0.76])
    cmap = plt.get_cmap("Blues")
    norm = mpl.colors.Normalize(vmin=LAT_CMAP_MS_MIN, vmax=LAT_CMAP_MS_MAX)
    cb = ColorbarBase(
        ax_cbar,
        cmap=cmap,
        norm=norm,
        orientation="vertical",
    )
    cb.set_label("Latency (ms)", fontsize=13, fontweight="bold")
    cb.ax.tick_params(labelsize=10)
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Data assembly ────────────────────────────────────────────────────────────


def build_heatmap_arrays(results, sm_list, l2_list):
    """
    Convert the flat list of result dicts into 2D arrays
    (shape: nL2 × nSM) for latency and power.

    Only latency-passing entries get real values; everything else is masked.
    X-axis = SM count (ascending left → right, index 0 = smallest SM).
    Y-axis = L2 size  (descending top → bottom, index 0 = largest L2).
    """
    sm_arr = sorted(set(sm_list))  # smallest SM at col 0 (left)
    l2_arr = sorted(set(l2_list), reverse=True)  # largest L2 at row 0 (top)

    lat_data = np.full((len(l2_arr), len(sm_arr)), np.nan)
    pwr_data = np.full((len(l2_arr), len(sm_arr)), np.nan)

    sm_idx = {v: i for i, v in enumerate(sm_arr)}
    l2_idx = {v: i for i, v in enumerate(l2_arr)}

    for r in results:
        if not r.get("lat_ok", False):
            continue
        si = sm_idx.get(r["sm_count"])
        li = l2_idx.get(r["l2_mb"])
        if si is None or li is None:
            continue
        lat_data[li, si] = r["total_lat_ms"]
        pwr_data[li, si] = r["avg_power_w"] * CHIP_NUM

    # Drop SM columns where every L2 entry is NaN (area-fail or fully pruned).
    # A column is kept only if at least one L2 size yields a valid result.
    keep_cols = ~np.all(np.isnan(lat_data), axis=0)
    sm_arr = [sm for sm, keep in zip(sm_arr, keep_cols) if keep]
    lat_data = lat_data[:, keep_cols]
    pwr_data = pwr_data[:, keep_cols]

    # When no config passes, keep a single all-NaN column so the caller can
    # still render a column of gray cells.  sm_arr is left empty to signal
    # the "no feasible" state to the plotting function.
    if len(sm_arr) == 0:
        lat_data = np.full((len(l2_arr), 1), np.nan)
        pwr_data = np.full((len(l2_arr), 1), np.nan)

    lat_masked = np.ma.masked_invalid(lat_data)
    pwr_masked = np.ma.masked_invalid(pwr_data)
    return lat_masked, pwr_masked, sm_arr, l2_arr


# ── CLI ──────────────────────────────────────────────────────────────────────

_MEM_FREQ_CHOICES = [6400, 8533, 9600, 10667, 12800]
_MEM_BITWIDTH_CHOICES = [128, 256]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run DSE grid search and visualize latency / power heatmaps "
            "(SM count × L2 cache size)."
        )
    )
    parser.add_argument(
        "--area",
        type=int,
        required=True,
        choices=[100, 200, 400],
        help="SoC area in mm² (100, 200 or 400).  GPU ≤ 35%% of total.",
    )
    parser.add_argument(
        "--base_hw",
        required=True,
        choices=["Orin", "Thor"],
        help="Base hardware platform (Orin or Thor).",
    )
    parser.add_argument(
        "--inference_config",
        required=True,
        choices=["Robo-S", "Robo-L", "AD-S", "AD-L"],
        help="Inference workload configuration.",
    )
    parser.add_argument(
        "--precision",
        required=True,
        help="LLM precision (Orin: fp16_int4 | int8; Thor: fp8 | fp4).",
    )
    parser.add_argument(
        "--latency_limit",
        type=float,
        default=110.0,
        help="End-to-end latency constraint in ms (default: 100 ms).",
    )
    parser.add_argument(
        "--mem_freq",
        type=int,
        choices=_MEM_FREQ_CHOICES,
        default=None,
        help="Memory frequency in MT/s (default: 6400 for Orin, 8533 for Thor).",
    )
    parser.add_argument(
        "--mem_bitwidth",
        type=int,
        choices=_MEM_BITWIDTH_CHOICES,
        default=None,
        help="Memory bus width in bits (default: 256).",
    )
    parser.add_argument(
        "--llm_model",
        default="Qwen3_8B",
        choices=list(_MODEL_DEFAULT_DEGREE.keys()),
        help="LLM model for the grid search (default: Qwen3_8B).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=("Directory to save PNG outputs (default: same folder as this script)."),
    )
    args = parser.parse_args()

    # ── Resolve defaults ────────────────────────────────────────────────────
    default_freq, default_bw = _DEFAULT_MEM[args.base_hw]
    mem_freq: int = args.mem_freq if args.mem_freq is not None else default_freq
    mem_bitwidth: int = (
        args.mem_bitwidth if args.mem_bitwidth is not None else default_bw
    )

    # ── Validate mem combination ────────────────────────────────────────────
    if mem_freq in _LPDDR5_FREQS and mem_bitwidth not in _LPDDR5_BITWIDTHS:
        parser.error(
            f"mem_freq={mem_freq} (LPDDR5/5x) requires mem_bitwidth in "
            f"{sorted(_LPDDR5_BITWIDTHS)}, got {mem_bitwidth}."
        )
    if mem_freq in _LPDDR6_FREQS and mem_bitwidth not in _LPDDR6_BITWIDTHS:
        parser.error(
            f"mem_freq={mem_freq} (LPDDR6) requires mem_bitwidth in "
            f"{sorted(_LPDDR6_BITWIDTHS)}, got {mem_bitwidth}."
        )

    # ── Validate precision ──────────────────────────────────────────────────
    valid_precisions = LLM_PRECISION_CHOICES[args.base_hw]
    if args.precision not in valid_precisions:
        parser.error(
            f"For {args.base_hw}, --precision must be one of {valid_precisions}. "
            f"Got: {args.precision!r}"
        )

    # ── Area / bitwidth guard ───────────────────────────────────────────────
    if args.area < 400 and mem_bitwidth in {256}:
        print(
            f"[ERROR] area={args.area} mm² (< 400) is not compatible with "
            f"mem_bitwidth={mem_bitwidth}-bit: not enough shoreline for PHYs."
        )
        sys.exit(1)

    if args.area < 400:
        ORIN_L2_MB_CANDIDATES.insert(0, 1.0)
        THOR_L2_MB_CANDIDATES.insert(0, 8.0)

    llm_model: str = args.llm_model
    llm_degree: int = _MODEL_DEFAULT_DEGREE[llm_model]

    out_dir = args.output_dir if args.output_dir else _THIS_DIR
    os.makedirs(out_dir, exist_ok=True)

    # ── Run DSE ─────────────────────────────────────────────────────────────
    results = run_dse(
        base_hw=args.base_hw,
        soc_area_mm2=float(args.area),
        inference_config=args.inference_config,
        llm_precision=args.precision,
        latency_limit_ms=args.latency_limit,
        mem_freq=mem_freq,
        mem_bitwidth=mem_bitwidth,
        llm_model=llm_model,
        llm_degree=llm_degree,
    )

    # ── Full candidate grids (all points, including area-pruned ones) ───────
    sm_candidates = ORIN_SM_CANDIDATES if args.base_hw == "Orin" else THOR_SM_CANDIDATES
    l2_candidates = (
        ORIN_L2_MB_CANDIDATES if args.base_hw == "Orin" else THOR_L2_MB_CANDIDATES
    )

    # ── Build masked 2D arrays ───────────────────────────────────────────────
    lat_masked, pwr_masked, sm_arr, l2_arr = build_heatmap_arrays(
        results, sm_candidates, l2_candidates
    )

    passing = [r for r in results if r.get("lat_ok", False)]
    print(
        f"\n  {len(passing)} / {len(results)} evaluated configs pass "
        f"the {args.latency_limit:.0f} ms latency limit."
    )

    # Short tag for filenames
    tag = (
        f"{args.base_hw}_{args.inference_config}_{llm_model}_"
        f"{args.precision}_{args.area}mm2_{mem_freq}MHz_{mem_bitwidth}bit_{args.latency_limit:.0f}ms"
    ).replace("/", "_")

    print("\n  Generating combined heatmap …")

    # ── Combined heatmap: colored by latency, annotated with latency + power ─
    plot_combined_heatmap(
        lat_masked,
        pwr_masked,
        sm_arr,
        l2_arr,
        out_path=os.path.join(out_dir, f"combined_heatmap_{tag}.png"),
    )

    cbar_path = os.path.join(out_dir, "latency_colorbar_20_100ms.png")
    print("\n  Generating standalone latency colorbar …")
    save_standalone_latency_colorbar(cbar_path)

    print("\n  Done.")


if __name__ == "__main__":
    main()
