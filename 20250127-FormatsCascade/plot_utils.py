"""Generate paper-ready plots"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import sys

PALETTE = sns.color_palette("Dark2")


def configure(disable_tex_for_debug_speed: bool = False) -> None:
    """Place at the start of the notebook, to set up defaults."""
    print(
        "Recommend (Ubuntu):\n"
        "  sudo apt-get install cm-super dvipng fonts-cmu texlive-latex-extra"
    )
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    sns.set_palette(PALETTE)
    font_name = "CMU Serif"
    matplotlib.rcParams.update(
        {
            # Fonts
            "font.family": "serif",
            "font.serif": [font_name],
            "text.usetex": not disable_tex_for_debug_speed,
            # General
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.edgecolor": "none",
            "legend.fontsize": "11",
            "lines.markersize": 3,
        }
    )
    try:
        matplotlib.font_manager.findfont(
            font_name, rebuild_if_missing=True, fallback_to_default=False
        )
    except ValueError as e:
        print(
            f"Couldn't find font {font_name!r}.\nOn Ubuntu:\n"
            "  sudo apt install fonts-cmu\n"
            "  rm ~/.cache/matplotlib/fontlist-*.json\n"
            "  (restart kernel)\n"
            f"  (original error: {e!r})"
        )


def save(name: str, push: bool = True) -> None:
    """Save and push a figure to the paper."""
    root = Path("overleaf/fig")
    if not root.exists():
        raise ValueError(
            f"Couldn't find {root} - please clone the paper into overleaf/"
        )
    plt.savefig(root / f"{name}.pdf", bbox_inches="tight")
    if push:
        for git_cmd in [
            "add fig/",
            "commit -m 'Update figures' --quiet",
            "pull --rebase --quiet",
            "push --quiet",
        ]:
            cmd = f"git -C overleaf {git_cmd}"
            # print(f"$ {cmd}", file=sys.stderr)
            if subprocess.call(cmd, shell=True):
                print(f"Error running {cmd} - aborting")
                return
