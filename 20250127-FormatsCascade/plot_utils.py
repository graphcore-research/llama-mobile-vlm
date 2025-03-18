"""Generate paper-ready plots"""

import inspect
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = sns.color_palette("Dark2")
SEQ_PALETTE = sns.color_palette("flare", as_cmap=True)

DISPLAY_NAMES = {
    "bits": "$b$",
    "block_size": "$B$",
    "qrmse_norm": "$R$",
    "LM": "Lloyd-Max",
}
CRP_LABEL = r"$\sqrt[3]{\mathrm{p}}$"


def display_name(t: str) -> str:
    if t in DISPLAY_NAMES:
        return DISPLAY_NAMES[t]
    if m := re.match(r"CRD-(N|L|T\[([\d.]+)\])-[RA]S", t):
        return (
            CRP_LABEL
            + dict(
                N=lambda: " Normal",
                L=lambda: " Laplace",
                T=lambda: f" t[$\\nu={m.group(2)}$]",
            )[m.group(1)[0]]()
        )
    return t


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
            # Latex
            "text.latex.preamble": "\n".join(
                [
                    r"\usepackage{amsmath}",
                    r"\usepackage{bm}",
                    r"\newcommand{\norm}[2]{\left \lVert #1 \right \rVert_{#2}}",
                ]
            ),
            # General
            "figure.figsize": (8, 3),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.edgecolor": "none",
            "legend.fontsize": "11",
            "axes.titlesize": "11",
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


def share_legend(figure: matplotlib.figure.Figure) -> None:
    handles, labels = figure.axes[00].get_legend_handles_labels()
    for ax in figure.axes:
        assert ax.get_legend_handles_labels()[1] == labels
        ax.legend_.remove()
    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))


def tidy(figure: matplotlib.figure.Figure) -> None:
    figure.tight_layout()

    for ax in figure.axes:
        for label in [ax.xaxis.label, ax.yaxis.label]:
            label.set_text(display_name(label.get_text()))

    for legend in filter(None, [ax.legend_ for ax in figure.axes] + figure.legends):
        title = legend.get_title()
        title.set_text(display_name(title.get_text()))
        for text in legend.get_texts():
            text.set_text(display_name(text.get_text()))


# Subplot grids


@dataclass
class Grid:
    df: pd.DataFrame
    row: str | None
    row_values: list[str]
    col: str | None
    col_values: list[str]
    axes: np.ndarray[matplotlib.axes.Axes]
    figure: matplotlib.figure.Figure

    def __iter__(self) -> Iterable[Any]:
        """Iterate through ((key,), DataFrame, Axes) tuples."""
        for row_value, axr in zip(self.row_values, self.axes):
            for col_value, ax in zip(self.col_values, axr):
                d, key = self.df, ()
                if self.row is not None:
                    key = (*key, row_value)
                    d = d[d[self.row] == row_value]
                if self.col is not None:
                    key = (*key, col_value)
                    d = d[d[self.col] == col_value]
                yield (key, d, ax)


def grid(
    df: pd.DataFrame,
    row: str | None = None,
    col: str | None = None,
    sharex: bool = False,
    sharey: bool = False,
) -> Grid:
    """Create a grid of matplotlib plots (much like seaborn, but plainer if not simpler)."""
    row_values = df[row].unique() if row else [None]
    col_values = df[col].unique() if col else [None]
    figw, figh = matplotlib.rcParams["figure.figsize"]
    figure, axes = plt.subplots(
        nrows=len(row_values),
        ncols=len(col_values),
        figsize=(figw, figh * len(row_values)),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    return Grid(
        df=df,
        row=row,
        row_values=row_values,
        col=col,
        col_values=col_values,
        axes=axes,
        figure=figure,
    )


# Paper sync


def push_to_paper() -> None:
    for git_cmd in [
        "add code/ fig/",
        "commit -m 'Update figures' --quiet",
        "pull --rebase --quiet",
        "push --quiet",
    ]:
        cmd = f"git -C overleaf {git_cmd}"
        # print(f"$ {cmd}", file=sys.stderr)
        if subprocess.call(cmd, shell=True):
            print(f"Error running {cmd!r} -- aborting")
            return


def save(name: str, push: bool = True) -> None:
    """Save and push a figure to the paper."""
    root = Path("overleaf/fig")
    if not root.exists():
        raise ValueError(
            f"Couldn't find {root} - please clone the paper into overleaf/"
        )
    plt.savefig(root / f"{name}.pdf", bbox_inches="tight")
    if push:
        push_to_paper()


def save_code(fn: Callable[..., Any], push: bool = True) -> None:
    body = inspect.getsource(fn).splitlines()[1:]
    body = [re.sub(r"^    ", "", x) for x in body]
    body = [x for x in body if "# IGNORE" not in x]
    code = "\n".join(body) + "\n"
    (Path("overleaf/code") / f"{fn.__name__}.py").write_text(code)
    if push:
        push_to_paper()
