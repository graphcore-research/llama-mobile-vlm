"""Generate paper-ready plots"""

import inspect
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import fractions
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PALETTE = sns.color_palette("Dark2")
SEQ_PALETTE = sns.color_palette("flare", as_cmap=True)

DISPLAY_NAMES = {
    "bits": "$b$",
    "block_size": "$B$",
    "qrmse_norm": "$R$",
    "LM": "Lloyd-Max",
    "rms": "RMS",
    # Formats
    "BFLOAT16": r"\texttt{bfloat16}",
    "EXP8": r"\texttt{E8M0}",
    "E0M3": r"INT4",
}
CRD_LABEL = r"$\sqrt[3]{\mathrm{p}}$"


def format_fraction(max_denominator: int = 10) -> Callable[[float, int], str]:
    def _format(x: float, n: int) -> str:
        if x == 0:
            return "0"
        f = fractions.Fraction.from_float(x).limit_denominator(max_denominator)
        if f.denominator == 1:
            return str(f.numerator)
        return f"$\\frac{{{f.numerator}}}{{{f.denominator}}}$"

    return _format


def drop_label(args: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in args.items() if k != "label"}


def display_name(t: str) -> str:
    if t in DISPLAY_NAMES:
        return DISPLAY_NAMES[t]
    if m := re.match(r"CRD-(N|L|T\[([\d.]+)\])-[RA]S", t):
        return (
            CRD_LABEL
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
            "legend.fontsize": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
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


def build_legend_handles(*groups_and_titles: list[dict[str, Any]] | str) -> list[Any]:
    handles = []
    sep = False
    for group_or_title in groups_and_titles:
        if sep:
            handles.append(matplotlib.patches.Patch(color="none"))
            sep = False
        if isinstance(group_or_title, str):
            handles.append(matplotlib.patches.Patch(color="none", label=group_or_title))
        else:
            for *_, args in group_or_title:
                args = dict(args)
                args.setdefault("color", "k")
                handles.append(matplotlib.lines.Line2D([], [], **args))
            sep = True
    return handles


def set_figure_legend(
    figure: matplotlib.figure.Figure,
    handles: Any = None,
    labels: Any = None,
    build: list[list[dict[str, Any]] | str] = None,
    loc: str = "center left",
    bbox_to_anchor: tuple[float, float] = (0.98, 0.5),
    **args: Any,
) -> None:
    if build is not None:
        assert handles is None and labels is None
        handles = build_legend_handles(*build)
    figure.legend(
        handles=handles, labels=labels, loc=loc, bbox_to_anchor=bbox_to_anchor, **args
    )
    if "left" in loc:
        extent = figure.legends[0].get_window_extent()
        figure.set_figwidth(2 * figure.get_figwidth() - extent.x1 / figure.dpi)


def share_legend(figure: matplotlib.figure.Figure, **args: Any) -> None:
    handles, labels = figure.axes[00].get_legend_handles_labels()
    for ax in figure.axes:
        assert ax.get_legend_handles_labels()[1] == labels
        if ax.legend_ is not None:
            ax.legend_.remove()
    set_figure_legend(figure, handles, labels, **args)


def tidy(figure: matplotlib.figure.Figure) -> None:
    figure.tight_layout()

    for ax in figure.axes:
        for label in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            label.set_text(display_name(label.get_text()))

    for legend in filter(None, [ax.legend_ for ax in figure.axes] + figure.legends):
        title = legend.get_title()
        title.set_text(display_name(title.get_text()))
        for text in legend.get_texts():
            text.set_text(display_name(text.get_text()))


# Subplot grids


@dataclass
class Grid:
    rows: list[str | None]
    cols: list[str | None]
    axes: np.ndarray[matplotlib.axes.Axes]
    figure: matplotlib.figure.Figure

    def __iter__(self) -> Iterable[Any]:
        """Iterate through ((key,), Axes) tuples."""
        for row, axr in zip(self.rows, self.axes):
            for col, ax in zip(self.cols, axr):
                key = ()
                if row is not None:
                    key = (*key, row)
                if col is not None:
                    key = (*key, col)
                yield (key, ax)


def grid(
    rows: list[str | None] = [None],
    cols: list[str | None] = [None],
    sharex: bool = False,
    sharey: bool = False,
) -> Grid:
    """Create a grid of matplotlib plots (much like seaborn, but plainer if not simpler)."""
    figw, figh = matplotlib.rcParams["figure.figsize"]
    figure, axes = plt.subplots(
        nrows=len(rows),
        ncols=len(cols),
        figsize=(figw, figh * len(rows)),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    return Grid(rows=rows, cols=cols, axes=axes, figure=figure)


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

    root = Path("overleaf/code")
    if not root.exists():
        raise ValueError(
            f"Couldn't find {root} - please clone the paper into overleaf/"
        )
    (root / f"{fn.__name__}.py").write_text(code)
    if push:
        push_to_paper()
