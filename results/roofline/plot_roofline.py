import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_element_based_roofline():
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "element_roofline.png"
    )

    configs = [
        {
            "name": "Orin INT8",
            "bw_gbps": 204.8,
            "compute_tops": 137.5,
            "bytes_per_elem": 1.0,
            "color": "#4F89BA",
            "linestyle": "-",
        },
        {
            "name": "Orin FP16",
            "bw_gbps": 204.8,
            "compute_tops": 68.75,
            "bytes_per_elem": 2.0,
            "color": "#A3C8EF",
            "linestyle": "-",
        },
        {
            "name": "Orin INT4",
            "bw_gbps": 204.8,
            "compute_tops": 68.75,
            "bytes_per_elem": 0.5,
            "color": "#CDE2FA",
            "linestyle": "--",
        },
        {
            "name": "Thor FP8",
            "bw_gbps": 273.15,
            "compute_tops": 517.5,
            "bytes_per_elem": 1.0,
            "color": "#D67F42",
            "linestyle": "-",
        },
        {
            "name": "Thor FP4",
            "bw_gbps": 273.15,
            "compute_tops": 1035.0,
            "bytes_per_elem": 0.5,
            "color": "#F2C28A",
            "linestyle": "-",
        },
    ]

    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(4, 2))

    bw_gelems = []
    bw_telems = []
    ridge_points = []
    for cfg in configs:
        bw_gelem = cfg["bw_gbps"] / cfg["bytes_per_elem"]
        bw_telem = bw_gelem / 1000.0
        ridge = cfg["compute_tops"] / bw_telem
        bw_gelems.append(bw_gelem)
        bw_telems.append(bw_telem)
        ridge_points.append(ridge)

    x_max = max(ridge_points) * 1.2
    y_max = max(cfg["compute_tops"] for cfg in configs) * 1.1
    x = np.linspace(0.0, x_max, 2500)

    for cfg, bw_gelem, bw_telem, ridge_point in zip(
        configs, bw_gelems, bw_telems, ridge_points
    ):
        y = np.minimum(cfg["compute_tops"], x * bw_telem)

        label = f"{cfg['name']} (BW: {bw_gelem:.1f} GElem/s)"
        ax.plot(
            x,
            y,
            label=label,
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=2.0,
        )

        ax.scatter(
            [ridge_point],
            [cfg["compute_tops"]],
            color=cfg["color"],
            s=18,
            zorder=5,
            edgecolor="white",
            linewidths=0.5,
        )

        if cfg["name"].startswith("Orin"):
            x_position = ridge_point + 0.1 * x_max
        else:
            x_position = ridge_point - 0.15 * x_max
        if cfg["name"].startswith("Orin") and cfg["name"].endswith("FP16"):
            y_position = cfg["compute_tops"] - 18
        else:
            y_position = cfg["compute_tops"] + 18
        if not cfg["name"].endswith("INT4"):
            ax.text(
                x_position,
                y_position,
                f"{cfg['compute_tops']} TOPS/TFLOPS",
                color="black",
                fontsize=7,
                fontweight="bold",
            )

    ax.set_xlabel(
        "Arithmetic Intensity (OPs/Element)",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_ylabel("Perf (TOPS/TFLOPS)", fontsize=9, fontweight="bold")
    # ax.set_title(
    #     "Element-based Roofline Model: NVIDIA AGX Orin vs Thor",
    #     fontsize=10,
    #     fontweight="bold",
    # )

    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, y_max)

    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="--",
        color="#B8B8B8",
        linewidth=0.7,
        alpha=0.95,
    )
    ax.grid(
        axis="x",
        linestyle="--",
        color="#D0D0D0",
        linewidth=0.5,
        alpha=0.85,
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)

    leg = ax.legend(
        loc="upper left",
        fontsize=7,
        frameon=True,
        fancybox=False,
        edgecolor="#888888",
        facecolor="white",
    )
    leg.get_frame().set_linewidth(0.8)

    ax.tick_params(axis="both", labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor="white")
    print(f"Chart successfully saved to: {out_path}")


if __name__ == "__main__":
    plot_element_based_roofline()
