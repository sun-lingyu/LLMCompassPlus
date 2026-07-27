#!/usr/bin/env python3
"""
DSE heatmap visualizer — single combined heatmap across L / M / S sizes.

Delegates all DSE computation to dse.dse.main(), then saves ONE PNG:

  heatmap_<tag>.png

A hardware configuration (SM count × L2 size) is only shown (colored /
annotated) when it is latency-feasible for ALL THREE sizes (L, M, S)
simultaneously — same "passes-all-sizes" notion as dse.dse's final summary.
For each such cell, the WORST CASE (max) latency across the three sizes is
displayed:

  - Cell color  → max(L, M, S) latency (ms), fixed scale 0-100 ms – Blues
                  colormap (no colorbar on the heatmap itself)
  - Cell label  → the max latency value (no unit), e.g. "98.0"
  Gray cells: configs that fail the area constraint, or fail the latency
  limit for at least one of the three sizes.

A standalone latency colorbar is also saved as a separate PNG.

Usage
-----
  python heatmap.py --area 400 --base_hw Orin \\
      --inference_config Robo --precision fp16_int4

  python heatmap.py --area 200 --base_hw Thor \\
      --inference_config AD --precision fp4 \\
      --mem_freq 8533 --mem_bitwidth 128 --latency_limit 100 --overlap
"""

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
    main as dse_main,
    ORIN_SM_CANDIDATES,
    ORIN_L2_MB_CANDIDATES,
    THOR_SM_CANDIDATES,
    THOR_L2_MB_CANDIDATES,
)


# ── Plotting constants ───────────────────────────────────────────────────────

# Latency colormap range (ms): heatmap + standalone colorbar share this scale.
LAT_CMAP_MS_MIN = 0.0
LAT_CMAP_MS_MAX = 100.0
LAT_CMAP_TICK_STEP_MS = 20.0

# Order in which sizes are compared / reported.
_SIZES = ("L", "M", "S")


# ── Plotting helpers ─────────────────────────────────────────────────────────


def _make_gray_cmap(base_cmap: str):
    """Return a copy of *base_cmap* that renders NaN cells in light gray."""
    cmap = plt.get_cmap(base_cmap).copy()
    cmap.set_bad(color="#cccccc")
    return cmap


def _annotate_cells_combined(ax, lat_masked, lat_vmin, lat_vmax):
    """Write the max latency across L/M/S in every non-masked cell, e.g. "98.0".

    Text color adapts to latency cell brightness (Blues colormap).
    """
    nrows, ncols = lat_masked.shape
    for i in range(nrows):
        for j in range(ncols):
            lat_val = lat_masked[i, j]
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
                i,
                f"{float(lat_val):.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )


def plot_combined_heatmap(
    lat_masked,
    sm_list,
    l2_list,
    *,
    out_path: str,
):
    """Draw a 2D heatmap colored by the worst-case (max) latency across L/M/S,
    annotated with that latency value.

    Latency color scale is fixed to LAT_CMAP_MS_MIN..LAT_CMAP_MS_MAX; no colorbar.

    lat_masked shape: (nL2, nSM)
    sm_list: SM candidates shown on X-axis (ascending left → right)
    l2_list: L2 candidates shown on Y-axis (descending top → bottom)
    """
    plt.style.use("seaborn-v0_8-white")

    no_feasible = len(sm_list) == 0
    nrows = len(l2_list)
    ncols = lat_masked.shape[1]

    cell_w = 0.35  # inches per SM column
    cell_h = 0.25  # inches per L2 row
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

    _annotate_cells_combined(ax, lat_masked, LAT_CMAP_MS_MIN, LAT_CMAP_MS_MAX)

    ax.set_xlabel("SM Count", fontsize=7, fontweight="bold")

    if no_feasible:
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
    """Save a vertical Blues colorbar for latency (ms), same scale as the heatmaps.

    Ticks are placed every LAT_CMAP_TICK_STEP_MS ms (e.g. 0, 20, 40, ..., 100).
    """
    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(0.35, 3.0))
    ax_cbar = fig.add_axes([0.22, 0.12, 0.38, 0.76])
    cmap = plt.get_cmap("Blues")
    norm = mpl.colors.Normalize(vmin=LAT_CMAP_MS_MIN, vmax=LAT_CMAP_MS_MAX)
    cb = ColorbarBase(ax_cbar, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_ticks(
        np.arange(LAT_CMAP_MS_MIN, LAT_CMAP_MS_MAX + 1e-9, LAT_CMAP_TICK_STEP_MS)
    )
    cb.set_label("Latency (ms)", fontsize=13, fontweight="bold")
    cb.ax.tick_params(labelsize=11)
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Data assembly ────────────────────────────────────────────────────────────


def _build_raw_latency_grid(results, sm_list, l2_list):
    """Convert a flat result list (one size) into a raw 2D latency array.

    Shape: (nL2, nSM), over the FULL sm_list/l2_list candidate grids (no
    column dropping here — that only happens once sizes are combined).

    X-axis = SM count (ascending left → right, index 0 = smallest SM).
    Y-axis = L2 size  (descending top → bottom, index 0 = largest L2).
    Only latency-passing entries receive real values; everything else NaN.
    """
    sm_arr = sorted(set(sm_list))
    l2_arr = sorted(set(l2_list), reverse=True)

    lat_data = np.full((len(l2_arr), len(sm_arr)), np.nan)

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

    return lat_data, sm_arr, l2_arr


def build_combined_heatmap_arrays(results_by_size, sm_list, l2_list, sizes=_SIZES):
    """Combine per-size DSE results into a single worst-case-latency grid.

    A cell is only populated (non-NaN) when EVERY size in *sizes* has a
    latency-passing result at that (SM, L2) pair.  Its value is the maximum
    latency across those sizes, and its label records which size produced
    that maximum ("L" / "M" / "S").

    Returns
    -------
    lat_masked : masked array (nL2, nSM) — max latency, NaN cells masked
    label_data : object array (nL2, nSM) — which size is the max (or None)
    sm_arr     : SM candidates that survive column-pruning
    l2_arr     : full L2 candidate list (descending)
    """
    grids = {}
    sm_arr = l2_arr = None
    for size in sizes:
        grid, sm_arr, l2_arr = _build_raw_latency_grid(
            results_by_size[size], sm_list, l2_list
        )
        grids[size] = grid

    stacked = np.stack([grids[size] for size in sizes], axis=0)  # (n_sizes, nL2, nSM)
    all_valid = ~np.isnan(stacked).any(axis=0)

    filled = np.where(np.isnan(stacked), -np.inf, stacked)
    max_idx = np.argmax(filled, axis=0)
    max_val = np.max(filled, axis=0)

    lat_data = np.where(all_valid, max_val, np.nan)

    label_data = np.full(max_idx.shape, None, dtype=object)
    for size_i, size in enumerate(sizes):
        label_data[all_valid & (max_idx == size_i)] = size

    # Drop SM columns where every L2 entry is NaN (area-fail / infeasible for
    # at least one size across the whole column).
    keep_cols = ~np.all(np.isnan(lat_data), axis=0)
    sm_arr = [sm for sm, keep in zip(sm_arr, keep_cols) if keep]
    lat_data = lat_data[:, keep_cols]
    label_data = label_data[:, keep_cols]

    # When no config passes for all sizes, keep a single all-NaN column so
    # the caller can still render a column of gray cells.  sm_arr is left
    # empty to signal the "no feasible" state to the plotting function.
    if len(sm_arr) == 0:
        lat_data = np.full((len(l2_arr), 1), np.nan)
        label_data = np.full((len(l2_arr), 1), None, dtype=object)

    lat_masked = np.ma.masked_invalid(lat_data)
    return lat_masked, label_data, sm_arr, l2_arr


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    results_by_size, _passing_keys, ctx = dse_main()

    base_hw = ctx["base_hw"]
    inference_config = ctx["inference_config"]
    sm_candidates = ORIN_SM_CANDIDATES if base_hw == "Orin" else THOR_SM_CANDIDATES
    l2_candidates = (
        ORIN_L2_MB_CANDIDATES if base_hw == "Orin" else THOR_L2_MB_CANDIDATES
    )

    tag = (
        f"{base_hw}_{inference_config}_{ctx['precision']}_"
        f"{ctx['area']}mm2_{ctx['mem_freq']}MHz_"
        f"{ctx['mem_bitwidth']}bit_{ctx['latency_limit']:.0f}ms"
        + ("_overlap" if ctx.get("overlap") else "")
    ).replace("/", "_")

    lat_masked, _label_data, sm_arr, l2_arr = build_combined_heatmap_arrays(
        results_by_size, sm_candidates, l2_candidates
    )

    n_feasible = int(np.sum(~np.ma.getmaskarray(lat_masked)))
    size_tags = "/".join(f"{inference_config}-{s}" for s in _SIZES)
    print(
        f"\n  {n_feasible} configs pass the {ctx['latency_limit']:.0f} ms limit "
        f"for ALL of {size_tags} simultaneously."
    )

    out_path = os.path.join(_THIS_DIR, f"heatmap_{tag}.png")
    print("\n  Generating combined heatmap (worst-case latency across L/M/S) …")
    plot_combined_heatmap(
        lat_masked,
        sm_arr,
        l2_arr,
        out_path=out_path,
    )

    print("\n  Generating standalone latency colorbar …")
    save_standalone_latency_colorbar(
        os.path.join(_THIS_DIR, "latency_colorbar_0_100ms.png")
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
