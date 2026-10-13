"""The figure comparing what the two training objectives build inside the network.

Both panels answer the same question from different directions. One reward function has a
reason to represent young and the other does not, so if anything about parenting is written
into these weights, it should be present in one family of policies and missing from the
other.

Left: silence hidden units one at a time, worst first, and watch pup seeking come apart. A
behaviour that is there falls towards chance. A behaviour that was never there starts at
chance and stays.

Right: blind the network to one sense at a time. This says which inputs the behaviour rests
on, which is a blunter question and a more readable answer.

Colours match the reward comparison figure, so the same objective is the same colour in both.
"""

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

FAMILIES = {
    "survival_policy": ("survival reward only", "#2a78d6"),
    "pen_policy": ("+ reward for offspring", "#eb6834"),
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2df", "#fcfcfb"
CHANCE = 0.25

# Only the pup senses are drawn. The probe that measures pup seeking puts the creature beside
# a pup with no food anywhere, so every food sense is already zero and blinding the network
# to one cannot change the number. A bar at zero there would be reporting how the probe was
# written rather than anything about the network.
SHOWN = ("pup_dx", "pup_dy", "pup_need")


def load(family):
    paths = sorted((ROOT / "results").glob(f"represent_{family}_s*.json"))
    if not paths:
        raise SystemExit(f"no representation results for {family}")
    return [json.load(open(p)) for p in paths]


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
    families = {name: load(name) for name in FAMILIES}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)

    ax = axes[0]
    ax.set_facecolor(SURFACE)
    for family, runs in families.items():
        label, colour = FAMILIES[family]
        curves = [[c["pup_seeking"] for c in r["progressive"]] for r in runs]
        steps = [c["removed"] for c in runs[0]["progressive"]]
        mean = [st.mean(v) for v in zip(*curves)]
        low = [min(v) for v in zip(*curves)]
        high = [max(v) for v in zip(*curves)]
        # The band is the full spread over three policies, not a confidence interval.
        ax.fill_between(steps, low, high, color=colour, alpha=0.15, linewidth=0)
        ax.plot(steps, mean, color=colour, linewidth=2.0, marker="o", markersize=3.5,
                label=label)
    ax.axhline(CHANCE, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate("chance", xy=(0, CHANCE), xytext=(2, 4), textcoords="offset points",
                color=MUTED, fontsize=8)
    _style(ax, "Silencing hidden units, worst first", "hidden units silenced",
           "fraction of moves towards the pup")

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    width = 0.34
    positions = range(len(SHOWN))
    for i, (family, runs) in enumerate(families.items()):
        label, colour = FAMILIES[family]
        costs = []
        for sense in SHOWN:
            per_run = []
            for r in runs:
                entry = next(s for s in r["sense_ablation"]["senses"] if s["sense"] == sense)
                per_run.append(entry["change"]["pup_seeking"])
            costs.append(st.mean(per_run))
        offset = (i - 0.5) * width
        ax.bar([p + offset for p in positions], costs, width, color=colour, label=label)
    ax.axhline(0, color=MUTED, linewidth=1.0)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(SHOWN, fontsize=9, color=MUTED)
    _style(ax, "Cost of blinding the network to a pup sense", "", "change in pup seeking")

    for ax in axes:
        legend = ax.legend(frameon=False, fontsize=8, loc="best")
        for text in legend.get_texts():
            text.set_color(MUTED)

    fig.suptitle("One reward term, and a representation that is either there or not",
                 fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ROOT / "results" / "representation.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")
    print()

    for family, runs in families.items():
        label, _ = FAMILIES[family]
        intact = runs[0]["ablation"]["intact"]
        print(f"{label} ({len(runs)} policies)")
        print("  intact  " + ", ".join(
            f"{k} {st.mean(r['ablation']['intact'][k] for r in runs):+.3f}" for k in intact))
        curve = [st.mean(v) for v in
                 zip(*[[c["pup_seeking"] for c in r["progressive"]] for r in runs])]
        print(f"  pup seeking, none silenced {curve[0]:.3f}, all but one silenced "
              f"{curve[-1]:.3f}")
        print("  blinding one sense costs, in pup seeking:")
        for sense in SHOWN:
            cost = st.mean(
                next(s for s in r["sense_ablation"]["senses"] if s["sense"] == sense)
                ["change"]["pup_seeking"] for r in runs)
            print(f"    {sense:<16s} {cost:+.3f}")
        print()


if __name__ == "__main__":
    main()
