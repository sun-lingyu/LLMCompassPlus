#!/usr/bin/env python3
"""
Parse nccl-tests .log files (2x Orin) in this directory, extract the
measured in-place latency for message sizes from 32B to 16M, compare
against an analytical latency model:

    T_comm = L0 + N * L + M / B

  L0 : per-collective fixed startup overhead, in microseconds
       (AllGather = 25 us, ReduceScatter = 25 us, AlltoAll = 35 us)
  N  : number of communication rounds actually run by the collective
       (with 2 ranks, every collective here completes in N = 1 round)
  L  : per-round latency, modeled as 4 us
  M  : data volume moved over a single link/direction for the collective
  B  : effective single-direction bandwidth, modeled as 4 GB/s (1 GB = 1e9 B,
       consistent with nccl-tests' own algbw/busbw GB/s convention)

For a 2-rank collective (AllGather / ReduceScatter / AllToAll), the
per-direction data volume is half of the "size (B)" column reported by
nccl-tests, i.e. M = size / 2:
  - AllGather:     each rank sends its size/2 chunk to the other rank.
  - ReduceScatter: each rank sends its size/2 chunk to be reduced by the other.
  - AllToAll:      with 2 ranks there is only 1 peer, each rank sends size/2 to it.

For each *.log file found here, this script produces a line+marker plot
(<Name>_comm_latency.png) with measured vs. modeled latency across message
sizes, annotated with the MAPE (mean absolute percentage error) of the model
relative to the measurement.
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# Model parameters
# ----------------------------------------------------------------------------
# Per-collective startup overhead L0 (us). Falls back to L0_DEFAULT with a
# warning if a *.log file's collective name isn't listed here.
L0_US_BY_COLLECTIVE = {
    "AllGather": 25.0,
    "ReduceScatter": 25.0,
    "AlltoAll": 35.0,
}
L0_DEFAULT_US = 25.0

N_ROUNDS = 1  # number of rounds actually run (2-rank collectives => 1)
L_US = 4.0  # per-round latency, in microseconds

B_GBPS = 4.0  # effective single-direction bandwidth, in GB/s (1 GB = 1e9 B)
B_BYTES_PER_US = B_GBPS * 1e9 / 1e6  # bytes / microsecond

SIZE_MIN = 32  # inclusive, bytes
SIZE_MAX = 16 * 1024 * 1024  # inclusive, bytes (16M)

# nccl-tests data row, e.g.:
#          32             4     float    none      -1    30.74    0.00    0.00      0    30.34    0.00    0.00      0
# columns: size count type redop root  time_oop algbw_oop busbw_oop wrong_oop  time_ip algbw_ip busbw_ip wrong_ip
ROW_RE = re.compile(
    r"^\s*(?P<size>\d+)\s+(?P<count>\d+)\s+(?P<type>\S+)\s+(?P<redop>\S+)\s+(?P<root>-?\d+)\s+"
    r"(?P<time_oop>[\d.]+)\s+(?P<algbw_oop>[\d.]+)\s+(?P<busbw_oop>[\d.]+)\s+(?P<wrong_oop>\S+)\s+"
    r"(?P<time_ip>[\d.]+)\s+(?P<algbw_ip>[\d.]+)\s+(?P<busbw_ip>[\d.]+)\s+(?P<wrong_ip>\S+)\s*$"
)


def parse_log(path):
    """Return sorted (size_bytes, measured_time_us) list using the in-place time."""
    data = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = ROW_RE.match(line)
            if not m:
                continue
            size = int(m.group("size"))
            time_ip_us = float(m.group("time_ip"))
            if SIZE_MIN <= size <= SIZE_MAX:
                # nccl-tests repeats warmup + measured rows for the same size;
                # keep the last occurrence (the actual measured iteration).
                data[size] = time_ip_us
    return sorted(data.items())


def model_time_us(size_bytes, name):
    """T_comm = L0 + N*L + M / B, with M = size_bytes / 2 (single-direction volume)."""
    l0_us = L0_US_BY_COLLECTIVE.get(name)
    if l0_us is None:
        print(
            f"[warn] no L0 configured for '{name}', falling back to {L0_DEFAULT_US} us"
        )
        l0_us = L0_DEFAULT_US
    m_bytes = np.asarray(size_bytes, dtype=float) / 2.0
    return l0_us + N_ROUNDS * L_US + m_bytes / B_BYTES_PER_US


def mape(measured, modeled):
    measured = np.asarray(measured, dtype=float)
    modeled = np.asarray(modeled, dtype=float)
    return float(np.mean(np.abs(modeled - measured) / measured) * 100.0)


def human_bytes(n):
    if n >= 1024 * 1024:
        return f"{n // (1024 * 1024)}M"
    if n >= 1024:
        return f"{n // 1024}K"
    return f"{n}"


def plot_one(name, sizes, measured_us, out_dir):
    sizes = np.asarray(sizes, dtype=float)
    measured_us = np.asarray(measured_us, dtype=float)
    modeled_us = model_time_us(sizes, name)
    err = mape(measured_us, modeled_us)
    l0_us = L0_US_BY_COLLECTIVE.get(name, L0_DEFAULT_US)

    measured_ms = measured_us / 1000.0
    modeled_ms = modeled_us / 1000.0

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(
        sizes, measured_ms, marker="o", label="Measured (in-place)", color="#1f77b4"
    )
    ax.plot(
        sizes, modeled_ms, marker="s", label="Model: $L_0 + NL + M/B$", color="#d62728"
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Message size M (Bytes)", fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontweight="bold")
    ax.set_title(
        f"{name}: measured vs. modeled communication latency\n"
        f"(2x Orin, $L_0$={l0_us:.0f} us, N={N_ROUNDS}, L={L_US:.0f} us, B={B_GBPS:.0f} GB/s)"
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    ax.text(
        0.98,
        0.05,
        f"MAPE = {err:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{name}_comm_latency.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path, err


def plot_combined(entries, out_dir, out_name, figsize=(2.5, 1.5)):
    """Compact combined plot: measured (solid) vs. model (dashed) for several
    collectives on one small figure. `entries` is a list of
    (name, sizes, measured_us) tuples.
    """
    colors = ["#4F89BA", "#F2C28A"]
    markers = ["s", "o"]  # hollow square, hollow circle

    # Match results/roofline/plot_roofline_orin_int8_thor_fp4.py styling/font.
    plt.style.use("seaborn-v0_8-white")

    fig, ax = plt.subplots(figsize=figsize)
    lines = []
    mape_lines = []
    legend_handles = []
    for i, (name, sizes, measured_us) in enumerate(entries):
        sizes = np.asarray(sizes, dtype=float)
        measured_us = np.asarray(measured_us, dtype=float)
        modeled_us = model_time_us(sizes, name)
        err = mape(measured_us, modeled_us)
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        measured_ms = measured_us / 1000.0
        modeled_ms = modeled_us / 1000.0

        # Measured: dashed line, no markers. Modeled: hollow markers only, no line.
        (l1,) = ax.plot(
            sizes,
            measured_ms,
            linestyle="--",
            linewidth=1.0,
            color=color,
            zorder=3,
        )
        (l2,) = ax.plot(
            sizes,
            modeled_ms,
            linestyle="None",
            marker=marker,
            markersize=3.5,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.0,
            zorder=3,
        )
        lines.extend([l1, l2])
        mape_lines.append(f"{name} {err:.1f}%")
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.0,
                markersize=5,
                label=name,
            )
        )

    ax.text(
        0.1,
        0.55,
        "MAPE:\n" + "\n".join(mape_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="black",
        fontweight="normal",
        zorder=4,
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(
        "Message Size (Bytes)",
        fontsize=10,
        fontweight="bold",
        ha="center",
    )
    ax.set_ylabel("Latency (ms)", fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4, linewidth=0.4)
    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=1,
        fontsize=9,
        frameon=True,
        fancybox=False,
        edgecolor="#888888",
        facecolor="white",
        handlelength=1.5,
        labelspacing=0.3,
        borderpad=0.3,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.set_zorder(1)
    for line in lines:
        line.set_zorder(3)

    fig.tight_layout(pad=0.4)
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    comm_dir = os.path.dirname(os.path.abspath(__file__))
    log_files = sorted(glob.glob(os.path.join(comm_dir, "*.log")))

    if not log_files:
        print(f"No .log files found in {comm_dir}")
        return

    all_data = {}
    print(f"{'Collective':<15}{'#points':>10}{'MAPE':>10}")
    for log_path in log_files:
        name = os.path.splitext(os.path.basename(log_path))[0]
        rows = parse_log(log_path)
        if not rows:
            print(f"[skip] {name}: no data rows parsed from {log_path}")
            continue
        sizes = [r[0] for r in rows]
        measured = [r[1] for r in rows]
        all_data[name] = (sizes, measured)
        out_path, err = plot_one(name, sizes, measured, comm_dir)
        print(f"{name:<15}{len(rows):>10}{err:>9.2f}%   -> {out_path}")

    combo_names = ["AllGather", "AlltoAll"]
    if all(n in all_data for n in combo_names):
        entries = [(n, *all_data[n]) for n in combo_names]
        out_path = plot_combined(
            entries, comm_dir, "AllGather_AlltoAll_comm_latency.png"
        )
        print(f"[combined] {'+'.join(combo_names)} -> {out_path}")
    else:
        missing = [n for n in combo_names if n not in all_data]
        print(f"[skip] combined plot: missing log(s) for {missing}")


if __name__ == "__main__":
    main()
