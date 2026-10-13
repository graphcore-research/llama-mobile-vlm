"""Generate assets/img/ste.svg: finite E2M1 quantisation and the identity STE.

E2M1 uses a sign bit, two exponent bits and one mantissa bit, including
subnormals and using every exponent encoding for finite values, as in the
vector-centroid notebook. Quantisation rounds to the nearest centroid.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Positive E2M1 centroids: subnormals, then three normal binades.
    positive = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    centroids = np.unique(np.concatenate((-positive, positive)))
    boundaries = (centroids[1:] + centroids[:-1]) / 2
    limit = 6.5

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "text.color": "#252530",
        "axes.labelcolor": "#252530",
        "xtick.color": "#62626d",
        "ytick.color": "#62626d",
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.96)
    ax.axhline(0, color="#dddddf", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#dddddf", linewidth=0.7, zorder=0)
    ax.stairs(
        centroids, np.concatenate(([-limit], boundaries, [limit])),
        baseline=None, color="#252530", linewidth=2, label="y = quantise(x)", zorder=3,
    )
    ax.plot(
        [-limit, limit], [-limit, limit], color="#64899f", linewidth=2,
        linestyle="--", label="y = x (surrogate for grad)", zorder=2,
    )
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="x", ylabel="y", aspect="equal")
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_yticks([-6, -3, 0, 3, 6])
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#85858f")
        spine.set_linewidth(0.7)
    ax.legend(loc="upper left", frameon=False, fontsize=12, handlelength=2)

    output = Path(__file__).resolve().parent.parent / "assets" / "img" / "ste.svg"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
