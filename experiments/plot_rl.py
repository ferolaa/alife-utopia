"""The figure comparing what the two training objectives produce.

Colours are fixed per objective and the two hues were checked for colour vision deficiency
separation, so the figure survives greyscale printing and colourblind readers. Each panel
has its own axis: the two measures are on different scales and putting them together would
invite comparisons between them that do not mean anything.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

COLOURS = {"survival": "#2a78d6", "offspring": "#eb6834"}
LABELS = {"survival": "survival reward only", "offspring": "+ reward for offspring"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"
CHANCE = 0.25


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def main():
    data = json.load(open(ROOT / "results" / "rl_curves.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)

    for ax, key, title, ylabel in (
        (axes[0], "pup", "Does it go to its young?", "fraction of moves towards the pup"),
        (axes[1], "births", "How much does it breed?", "births per episode"),
    ):
        ax.set_facecolor(SURFACE)
        for mode in ("survival", "offspring"):
            runs = [x for x in data if x["mode"] == mode]
            iters = [p["it"] for p in runs[0]["curve"]]
            series = [[p[key] for p in r["curve"]] for r in runs]
            mean = [sum(vals) / len(vals) for vals in zip(*series)]
            low = [min(vals) for vals in zip(*series)]
            high = [max(vals) for vals in zip(*series)]

            colour = COLOURS[mode]
            # The band is the full spread across seeds, not a confidence interval. With
            # five runs, showing every one of them is more honest than an error bar that
            # implies a distribution we have not established.
            ax.fill_between(iters, low, high, color=colour, alpha=0.15, linewidth=0)
            ax.plot(iters, mean, color=colour, linewidth=2.0, label=LABELS[mode])
            ax.annotate(LABELS[mode], xy=(iters[-1], mean[-1]), xytext=(5, 0),
                        textcoords="offset points", color=colour, fontsize=8, va="center")

        if key == "pup":
            ax.axhline(CHANCE, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
            ax.annotate("chance", xy=(iters[0], CHANCE), xytext=(2, 4),
                        textcoords="offset points", color=MUTED, fontsize=8)
        _style(ax, title, "training iteration", ylabel)
        legend = ax.legend(frameon=False, fontsize=8, loc="upper left")
        for text in legend.get_texts():
            text.set_color(MUTED)

    fig.suptitle(
        "Same network, same world. One extra reward term.",
        fontsize=13, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 0.93, 0.94))
    out = ROOT / "results" / "rl_comparison.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
