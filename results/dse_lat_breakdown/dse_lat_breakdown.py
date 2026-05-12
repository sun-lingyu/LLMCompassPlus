#!/usr/bin/env python3
"""
DSE latency breakdown bar chart.

Calls dse.dse.main() to run the full grid search, then picks the passing
hardware configuration with the smallest GPU die area and plots a stacked
bar chart showing the per-size latency breakdown.

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
  python dse_breakdown.py --area 400 --base_hw Orin \\
      --inference_config Robo --precision fp16_int4

  python dse_breakdown.py --area 200 --base_hw Thor \\
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

from dse.dse import (  # noqa: E402, I001
    main as dse_main,
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


# ── Helpers ──────────────────────────────────────────────────────────────────


def _pick_min_area_config(passing_keys: set, results_by_size: dict) -> tuple:
    """Return the (sm_count, l2_mb) key from *passing_keys* with the smallest
    GPU die area.  Area is read from the Size-L result dict (it is identical
    across all sizes for the same hardware config).
    """
    area_map = {
        (r["sm_count"], r["l2_mb"]): r["gpu_area_mm2"] for r in results_by_size["L"]
    }
    return min(passing_keys, key=lambda k: area_map.get(k, float("inf")))


def _get_breakdown(results: list, sm: int, l2: float) -> dict:
    """Return the result dict for the given (sm, l2) config, or None."""
    for r in results:
        if r["sm_count"] == sm and r["l2_mb"] == l2:
            return r
    return None


# ── Plot ─────────────────────────────────────────────────────────────────────


def plot_breakdown_bar(
    breakdown_by_size: dict,
    inference_config: str,
    sm: int,
    l2: float,
    gpu_area: float,
    latency_limit: float,
    *,
    out_path: str,
):
    """Draw a stacked bar chart of latency breakdown for the chosen config.

    breakdown_by_size: {"L": {"vit": ms, "prefill": ms, "decode": ms}, ...}
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

    # Annotate FPS above each bar
    for xi, total_ms in zip(x, bottoms):
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0
        ax.text(
            xi,
            total_ms + 0.8,
            f"{fps:.1f} FPS",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
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
    ax.set_ylim(bottom=0, top=180)
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
        print("\n  No passing configuration found — breakdown chart skipped.")
        return

    inference_config = ctx["inference_config"]
    latency_limit = ctx["latency_limit"]

    # Pick the passing config with the smallest GPU die area.
    sm, l2 = _pick_min_area_config(passing_keys, results_by_size)
    area_map = {
        (r["sm_count"], r["l2_mb"]): r["gpu_area_mm2"] for r in results_by_size["L"]
    }
    gpu_area = area_map[(sm, l2)]
    print(
        f"\n  Min-area passing config: SM={sm}  L2={l2:.1f} MB  area={gpu_area:.1f} mm²"
    )

    # Collect latency breakdown for each size.
    breakdown_by_size = {}
    for size in ("L", "M", "S"):
        r = _get_breakdown(results_by_size[size], sm, l2)
        if r is None:
            print(
                f"  [WARNING] No result found for size {size} at SM={sm} L2={l2} — skipping."
            )
            return
        breakdown_by_size[size] = {
            "vit": r.get("vit_lat_ms", 0.0),
            "prefill": r.get("llm_prefill_lat_ms", 0.0),
            "decode": r.get("llm_decode_lat_ms", 0.0),
        }

    tag = (
        f"{ctx['base_hw']}_{inference_config}_{ctx['precision']}_"
        f"{ctx['area']}mm2_{ctx['mem_freq']}MHz_"
        f"{ctx['mem_bitwidth']}bit_{ctx['latency_limit']:.0f}ms"
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
        out_path=out_path,
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
