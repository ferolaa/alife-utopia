"""What actually stops the population growing.

The colony settles around a hundred and twenty creatures in a pen with room for three
thousand. Calling that a density effect is easy and not yet earned, because a population can
stop growing for dull reasons. So this separates the candidates.

Every tick of every baseline run is sorted into a bucket by how many creatures were alive at
the time, and within each bucket the rates are worked out per creature per tick. Rates, not
totals: a bucket the colony spends longer in accumulates more of everything, and totals would
just be measuring how long it sat there.

Three things could be holding the ceiling down.

  food      if creatures are starving faster as the pen fills, food is the constraint
  nests     if the nest boxes run out, that is the constraint
  density   if creatures breed less as the pen fills while neither of the above holds,
            then what is falling is reproduction itself

The last one is Calhoun's claim, and it is the only one of the three that says anything
interesting. The other two have to be ruled out for it to stand.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

BUCKET = 20                  # population range per bucket
MIN_TICKS = 100              # buckets thinner than this are too noisy to plot

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2df", "#fcfcfb"
BIRTHS, DEATHS, NESTS = "#eb6834", "#2a78d6", "#1baf7a"


def gather(paths):
    """Per creature per tick rates, bucketed by how full the pen was."""
    buckets = defaultdict(lambda: defaultdict(float))
    for path in paths:
        for record in json.load(open(path)):
            population = record["population"]
            if population == 0:
                continue
            b = buckets[(population // BUCKET) * BUCKET]
            b["ticks"] += 1
            b["creature_ticks"] += population
            for key in ("births", "died_starving", "died_neglected", "died_old"):
                b[key] += record.get(key, 0)
            b["nests_free"] += record.get("nests_free", 0)

    rows = []
    for low in sorted(buckets):
        b = buckets[low]
        if b["ticks"] < MIN_TICKS:
            continue
        per = lambda k: 1000 * b[k] / b["creature_ticks"]   # noqa: E731
        rows.append({
            "population": low + BUCKET // 2,
            "ticks": int(b["ticks"]),
            "births": per("births"),
            "starvation": per("died_starving"),
            "neglect": per("died_neglected"),
            "old_age": per("died_old"),
            "nests_free": b["nests_free"] / b["ticks"],
        })
    return rows


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


def figure(rows, out):
    x = [r["population"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)

    ax = axes[0]
    ax.set_facecolor(SURFACE)
    ax.plot(x, [r["births"] for r in rows], color=BIRTHS, linewidth=2.0, marker="o",
            markersize=4, label="births")
    ax.plot(x, [r["starvation"] for r in rows], color=DEATHS, linewidth=2.0, marker="o",
            markersize=4, label="starvation deaths")
    _style(ax, "Births fall as the pen fills. Starvation does not rise.",
           "creatures alive", "per 1000 creature ticks")

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    ax.plot(x, [r["nests_free"] for r in rows], color=NESTS, linewidth=2.0, marker="o",
            markersize=4, label="nest boxes standing empty")
    ax.set_ylim(0, None)
    _style(ax, "And the nest boxes never run out.", "creatures alive", "empty nest boxes")

    for ax in axes:
        legend = ax.legend(frameon=False, fontsize=8, loc="best")
        for text in legend.get_texts():
            text.set_color(MUTED)

    fig.suptitle("What stops the population growing", fontsize=13, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else "campaign_shared_s*.json"
    paths = sorted((ROOT / "results").glob(pattern))
    if not paths:
        raise SystemExit(f"no runs matching {pattern}")

    rows = gather(paths)
    json.dump(rows, open(ROOT / "results" / "density.json", "w"), indent=1)
    figure(rows, ROOT / "results" / "density.png")

    print(f"{len(paths)} runs, {sum(r['ticks'] for r in rows)} ticks")
    print()
    print("creatures   ticks   births   starvation   neglect   old age   empty nests")
    for r in rows:
        print(f"{r['population']:9d} {r['ticks']:7d} {r['births']:8.2f} "
              f"{r['starvation']:12.2f} {r['neglect']:9.2f} {r['old_age']:9.2f} "
              f"{r['nests_free']:13.1f}")
    print()
    print("rates are per 1000 creature ticks")
    first, last = rows[0], rows[-1]
    drop = 100 * (1 - last["births"] / first["births"])
    print(f"births fall {drop:.0f} percent from the smallest bucket to the largest")
