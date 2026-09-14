"""Turning saved runs into the figures that go in the report.

The colours are fixed per condition and never reused for anything else, so the same
condition is the same colour in every figure. The three chosen hues were checked for
colour vision deficiency separation, so the figures stay readable in greyscale print and
for colourblind readers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # render to a file, with no window, so this runs anywhere
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# One fixed colour per condition. Assigned by identity, never by order of plotting.
CONDITION_COLOURS = {
    "baseline": "#2a78d6",   # blue
    "calhoun": "#eb6834",    # orange
    "freedman": "#1baf7a",   # aqua
}
FALLBACK_COLOUR = "#52514e"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e3e2df"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Recessive axes and grid, so the data is the loudest thing on the chart."""
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def _colour(name: str) -> str:
    return CONDITION_COLOURS.get(name, FALLBACK_COLOUR)


def _smooth(values, window: int = 25):
    """A simple moving average. Tick level numbers are noisy, and the shape is the point.

    The raw series is always drawn underneath at low opacity, so smoothing never hides
    what actually happened.
    """
    if window <= 1 or len(values) < window:
        return list(values)
    out, running = [], 0.0
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
        out.append(running / min(i + 1, window))
    return out


def plot_run(history: list[dict], name: str, path: Path | None = None) -> Path:
    """Four panels describing a single run.

    Each panel has its own axis. Two measures of different scale are never put on one
    pair of axes, because a shared axis invites false comparisons between them.
    """
    ticks = [r["tick"] for r in history]
    colour = _colour(name)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), facecolor=SURFACE)
    fig.suptitle(f"Run: {name}", fontsize=13, color=INK, x=0.02, ha="left")

    panels = [
        ("population", "Population", "creatures alive"),
        ("mean_neighbours", "Crowding actually experienced", "neighbours per creature"),
        ("mean_energy", "Mean energy", "energy"),
        ("food", "Food available", "items on the grid"),
    ]

    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        raw = [r[key] for r in history]
        ax.plot(ticks, raw, color=colour, linewidth=0.8, alpha=0.25)
        ax.plot(ticks, _smooth(raw), color=colour, linewidth=2.0)
        _style(ax, title, "tick", ylabel)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path or RESULTS_DIR / f"{name}_run.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_comparison(histories: dict[str, list[dict]], path: Path | None = None) -> Path:
    """The headline figure. All conditions on the same axes, one panel per measure.

    Every panel carries a legend, and each line is also labelled directly at its right
    hand end, so the conditions are never told apart by colour alone.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE)

    panels = [
        ("population", "Population over time", "creatures alive"),
        ("mean_neighbours", "Crowding experienced", "neighbours per creature"),
        ("births", "Births", "births per tick"),
    ]

    for ax, (key, title, ylabel) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for name, history in histories.items():
            ticks = [r["tick"] for r in history]
            raw = [r[key] for r in history]
            smoothed = _smooth(raw, window=50)
            colour = _colour(name)
            ax.plot(ticks, raw, color=colour, linewidth=0.7, alpha=0.18)
            ax.plot(ticks, smoothed, color=colour, linewidth=2.0, label=name)
            ax.annotate(
                name, xy=(ticks[-1], smoothed[-1]), xytext=(4, 0),
                textcoords="offset points", color=colour, fontsize=8,
                va="center", fontweight="medium",
            )
        _style(ax, title, "tick", ylabel)
        legend = ax.legend(frameon=False, fontsize=8, loc="upper left")
        for text in legend.get_texts():
            text.set_color(MUTED)

    fig.tight_layout()
    out = Path(path or RESULTS_DIR / "comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out
