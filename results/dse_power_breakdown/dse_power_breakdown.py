#!/usr/bin/env python3
"""
DSE power breakdown bar chart.

Calls dse.dse.main() to run the full grid search, picks the passing
hardware configuration with the smallest GPU die area, and plots a stacked
bar chart showing the **total system** power breakdown for each workload size.

Each bar is stacked as:
  Chip  (bottom) — SoC compute power × degree
  DRAM  (top)    — memory power × degree

"degree" is the number of SoCs used (L=4, M=2, S=1), so the bar height
represents the full multi-chip system power in watts.

Output
------
  power_breakdown_<tag>.png   saved in the same directory as this script.

Usage
-----
  python dse_power_breakdown.py --area 400 --base_hw Orin \\
      --inference_config Robo --precision fp16_int4

  python dse_power_breakdown.py --area 200 --base_hw Thor \\
      --inference_config AD --precision fp8 \\
      --mem_freq 10667 --mem_bitwidth 128 --latency_limit 120
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# ── Make the project root importable ────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dse.dse import main as dse_main  # noqa: E402, I001

# ── Plot constants ───────────────────────────────────────────────────────────

_SEG_COLORS = {
    "chip": "#4F89BA",  # TP dark blue  (matches latency breakdown palette)
    "dram": "#A3C8EF",  # TP light blue
}
_SEG_LABELS = {
    "chip": "Chip (SoC)",
    "dram": "DRAM",
}

_SIZE_DEGREE = {"L": 4, "M": 2, "S": 1}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _pick_min_area_config(passing_keys: set, results_by_size: dict) -> tuple:
    """Return the (sm_count, l2_mb) key with the smallest GPU die area."""
    area_map = {
        (r["sm_count"], r["l2_mb"]): r["gpu_area_mm2"] for r in results_by_size["L"]
    }
    return min(passing_keys, key=lambda k: area_map.get(k, float("inf")))


def _get_result(results: list, sm: int, l2: float) -> dict:
    """Return the result dict for the given (sm, l2) config, or None."""
    for r in results:
        if r["sm_count"] == sm and r["l2_mb"] == l2:
            return r
    return None


# ── Plot ─────────────────────────────────────────────────────────────────────


def plot_power_breakdown_bar(
    power_by_size: dict,
    inference_config: str,
    sm: int,
    l2: float,
    gpu_area: float,
    *,
    peak_by_size: dict = None,
    out_path: str,
):
    """Draw a stacked bar chart of total system power breakdown.

    power_by_size: {"L": {"chip": W, "dram": W}, "M": {...}, "S": {...}}
    peak_by_size:  {"L": peak_W, "M": peak_W, "S": peak_W}  (optional, system-level)
    Values are already multiplied by degree (full system power).
    """
    plt.style.use("seaborn-v0_8-white")

    sizes = ["S", "M", "L"]
    segments = ["dram", "chip"]

    fig, ax = plt.subplots(figsize=(2.5, 2))

    x = np.arange(len(sizes)) * 0.5
    bar_w = 0.25
    bottoms = np.zeros(len(sizes))
    margin = bar_w * 1.5

    for seg in segments:
        vals = np.array([power_by_size[s][seg] for s in sizes])
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
            if val < 1.0:
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

    # Annotate total (avg) and peak above each bar — method C
    for xi, total_w, size in zip(x, bottoms, sizes):
        if peak_by_size is not None:
            peak_w = peak_by_size[size]
            label = f"{total_w:.1f} W\n(peak {peak_w:.1f} W)"
        else:
            label = f"{total_w:.1f} W"
        ax.text(
            xi,
            total_w + 0.5,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
            zorder=6,
            linespacing=1.15,
        )

    degree_labels = {s: _SIZE_DEGREE[s] for s in sizes}
    x_labels = [f"{inference_config}-{s}\n({degree_labels[s]}-Chip)" for s in sizes]
    ax.set_xlim(x[0] - margin, x[-1] + margin)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)

    ax.set_ylabel("Average Power (W)", fontsize=9, fontweight="bold")
    ax.set_ylim(bottom=0, top=150)
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
        fontsize=7,
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="#888888",
        facecolor="white",
        labelspacing=0.2,
    )
    leg.set_zorder(5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    results_by_size, passing_keys, ctx = dse_main()

    if not passing_keys:
        print("\n  No passing configuration found — power breakdown chart skipped.")
        return

    inference_config = ctx["inference_config"]

    # Pick the passing config with the smallest GPU die area.
    sm, l2 = _pick_min_area_config(passing_keys, results_by_size)
    area_map = {
        (r["sm_count"], r["l2_mb"]): r["gpu_area_mm2"] for r in results_by_size["L"]
    }
    gpu_area = area_map[(sm, l2)]
    print(
        f"\n  Min-area passing config: SM={sm}  L2={l2:.1f} MB  area={gpu_area:.1f} mm²"
    )

    # Collect power breakdown for each size and multiply by degree.
    power_by_size = {}
    peak_by_size = {}
    for size in ("L", "M", "S"):
        r = _get_result(results_by_size[size], sm, l2)
        if r is None:
            print(
                f"  [WARNING] No result found for size {size} at SM={sm} L2={l2} — skipping."
            )
            return
        degree = _SIZE_DEGREE[size]
        chip_per_card = r.get("chip_power_w", r["avg_power_w"])
        dram_per_card = r.get("dram_power_w", 0.0)
        peak_per_card = r.get("peak_power_w", r["avg_power_w"])
        power_by_size[size] = {
            "chip": round(chip_per_card * degree, 2),
            "dram": round(dram_per_card * degree, 2),
        }
        peak_by_size[size] = round(peak_per_card * degree, 2)
        print(
            f"  Size {size} (×{degree}): chip={chip_per_card:.1f} W/card  "
            f"dram={dram_per_card:.1f} W/card  peak={peak_per_card:.1f} W/card  "
            f"→ system chip={power_by_size[size]['chip']:.1f} W  "
            f"dram={power_by_size[size]['dram']:.1f} W  "
            f"peak={peak_by_size[size]:.1f} W"
        )

    tag = (
        f"{ctx['base_hw']}_{inference_config}_{ctx['precision']}_"
        f"{ctx['area']}mm2_{ctx['mem_freq']}MHz_"
        f"{ctx['mem_bitwidth']}bit_{ctx['latency_limit']:.0f}ms"
    ).replace("/", "_")

    out_path = os.path.join(_THIS_DIR, f"power_breakdown_{tag}.png")
    print("\n  Generating power breakdown bar chart …")
    plot_power_breakdown_bar(
        power_by_size,
        inference_config,
        sm,
        l2,
        gpu_area,
        peak_by_size=peak_by_size,
        out_path=out_path,
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
