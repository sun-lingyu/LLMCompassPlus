import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEVICE_CONFIG = {
    "Thor": {"precisions": {"prefill": ["fp8", "fp4"], "decode": ["fp8", "fp4"]}},
    "Orin": {"precisions": {"prefill": ["fp16", "int8"], "decode": ["int4", "int8"]}},
}

MODELS = ["Qwen3_1.7B", "Qwen3_4B", "Qwen3_8B"]
DEGREES = [1, 2, 4]


def build_filename(device, precision, phase, seq_len, spec_tokens):
    if spec_tokens == "-1":
        return f"{device}_{precision}_{phase}_{seq_len}_power.csv"
    if precision == "fp4" and phase == "decode" and int(spec_tokens) <= 128:
        spec_tokens = 128
    return f"{device}_{precision}_{phase}_{seq_len}_{spec_tokens}_power.csv"


def load_heatmap_data(device, strategy, phase, seq_len, spec_tokens):
    cfg = DEVICE_CONFIG[device]
    precisions = cfg["precisions"][phase]

    # heatmap shape: (3 rows=degrees, 6 cols=models x precisions)
    data = np.full((len(DEGREES), len(MODELS) * len(precisions)), np.nan)

    for pi, prec in enumerate(precisions):
        filename = build_filename(device, prec, phase, seq_len, spec_tokens)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Cannot find CSV file: {filename}")
        df = pd.read_csv(filename)
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()

        for di, deg in enumerate(DEGREES):
            if deg == 1:
                mask = df.iloc[:, 0] == "No Parallelism"
            else:
                mask = df.iloc[:, 0] == f"{strategy}={deg}"

            matched = df[mask]
            if matched.empty:
                continue

            row = matched.iloc[0]
            for mi, model in enumerate(MODELS):
                if model in df.columns:
                    col_idx = pi * len(MODELS) + mi
                    data[di, col_idx] = float(row[model])

    return data, precisions


def main():
    parser = argparse.ArgumentParser(description="Generate Power Heatmap")
    parser.add_argument(
        "--device", type=str, required=True, help="Device name (e.g., Thor, Orin)"
    )
    parser.add_argument(
        "--phase", type=str, required=True, help="Phase (e.g., prefill, decode)"
    )
    parser.add_argument(
        "--seq_len", type=str, required=True, help="Sequence length (e.g., 1024)"
    )
    parser.add_argument(
        "--spec_tokens",
        type=str,
        required=False,
        default="-1",
        help="Number of special tokens (e.g., 64), -1 means none",
    )
    parser.add_argument(
        "--precision",
        type=str,
        required=False,
        default=None,
        help="(Unused for heatmap; both precisions are loaded automatically)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Parallelism strategy (TP or CP)",
    )
    args = parser.parse_args()

    if args.device not in DEVICE_CONFIG:
        raise ValueError(
            f"Unknown device '{args.device}'. Supported: {list(DEVICE_CONFIG.keys())}"
        )

    strategy = args.strategy.upper()
    data, precisions = load_heatmap_data(
        args.device, strategy, args.phase, args.seq_len, args.spec_tokens
    )

    # --- Column labels: precision x model (grouped by precision) ---
    col_labels = []
    for prec in precisions:
        for model in MODELS:
            col_labels.append(f"{model.replace('Qwen3_', '')}")

    # --- Row labels: strategy=degree ---
    row_labels = ["No Para" if d == 1 else f"{strategy}={d}" for d in DEGREES]

    # --- Plot ---
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(2.8, 1.5))

    vmin, vmax = 20, 100
    cmap = "YlOrRd"
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Annotate each cell with per-device power, and total power for degree > 1
    for i, deg in enumerate(DEGREES):
        for j in range(len(MODELS) * len(precisions)):
            val = data[i, j]
            if np.isnan(val):
                continue
            normalized = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            text_color = "white" if normalized > 0.55 else "black"
            if deg == 1:
                ax.text(
                    j,
                    i,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=text_color,
                )
            else:
                total = val * deg
                ax.text(
                    j,
                    i - 0.1,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=text_color,
                )
                ax.text(
                    j,
                    i + 0.25,
                    f"({total:.0f})",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=text_color,
                    fontstyle="italic",
                )

    # Axis ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)

    # Vertical separators between precision groups
    n_models = len(MODELS)
    for g in range(1, len(precisions)):
        ax.axvline(g * n_models - 0.5, color="white", linewidth=2.5, zorder=3)

    # Add top-level precision group labels via secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    group_centers = [
        (g * n_models + (n_models - 1) / 2) for g in range(len(precisions))
    ]
    ax2.set_xticks(group_centers)
    ax2.set_xticklabels(
        precisions,
        fontsize=7,
        fontweight="bold",
        fontstyle="italic",
    )
    ax2.tick_params(length=0, pad=1)

    # Grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DEGREES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=1.5)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)

    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)

    plt.tight_layout()

    # Output filename
    if args.spec_tokens == "-1":
        out_filename = f"{args.device}_{args.phase}_{args.strategy}_{args.seq_len}_power_heatmap.png"
    else:
        out_filename = f"{args.device}_{args.phase}_{args.strategy}_{args.seq_len}_{args.spec_tokens}_power_heatmap.png"

    plt.savefig(out_filename, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"Heatmap saved to: {out_filename}")

    # --- Standalone colorbar ---
    fig_cb, ax_cb = plt.subplots(figsize=(0.2, 1.5))
    fig_cb.subplots_adjust(left=0.3, right=0.55, top=0.97, bottom=0.03)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig_cb.colorbar(sm, cax=ax_cb)
    cbar.set_label("Power (W)", fontsize=7, fontweight="bold")
    cbar.ax.tick_params(labelsize=6)
    cb_filename = "power_colorbar.png"
    fig_cb.savefig(cb_filename, dpi=1200, facecolor="white", bbox_inches="tight")
    print(f"Colorbar saved to: {cb_filename}")


if __name__ == "__main__":
    main()
