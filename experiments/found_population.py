"""Produce the population that every experiment starts from.

This is setup, not an experiment. It runs the full model from random brains until a
population establishes itself, then saves the survivors' brains for the real experiments to
use.

Founding from random brains is unreliable if left alone: a population starting with no idea
how to raise young loses most of its members while evolution discovers parental care, and
roughly a third of attempts never recover. That is a genuinely interesting fact about
evolving care from nothing, but it is not the question this project asks, and if every
experimental run had to survive it first, the results would mostly record which runs got
lucky. So the lottery is run once, here, and the winner is reused everywhere.

Usage:
    python3 experiments/found_population.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import metrics  # noqa: E402
import simulation  # noqa: E402
from config import DEFAULT  # noqa: E402
from evaluate import summarise_population  # noqa: E402
from population import save_brains  # noqa: E402

# The founders evolve under the full model's own conditions. Founding them somewhere
# gentler would leave them adapted to a world they are then taken out of, and the dip that
# followed would have nothing to do with anything we want to measure.
FOUNDING = DEFAULT.variant(
    "founding",
    n_ticks=8000,
    n_initial_agents=400,
    max_population=2000,
    ageing_enabled=True,
    food_unlimited=True,
    n_food=300,
    nests_enabled=True,
    n_nests=60,
    parental_care_enabled=True,
    # Crowding stress is on here for the same reason everything else is: the founders must
    # be adapted to the world the experiments actually run in. Founding them in a world
    # where crowding is free and then making it costly would produce a crash caused by the
    # switch itself, which would be easy to mistake for a crowding collapse.
    crowding_cost_enabled=True,
)

# What counts as established. A run limping along with a handful of survivors has not
# produced a population worth founding anything on.
MINIMUM_POPULATION = 300


def main(max_attempts: int = 6) -> int:
    for seed in range(max_attempts):
        cfg = FOUNDING.variant(f"founding_seed{seed}", seed=seed)
        print(f"attempt {seed}: running {cfg.n_ticks} ticks...", flush=True)
        result = simulation.run(cfg)
        summary = metrics.summarise(result)
        final = summary["final_population"]
        print(
            f"  finished in {result['seconds']}s: peak {summary['peak_population']}, "
            f"final {final}, extinct at {summary['extinct_at']}",
            flush=True,
        )

        if final < MINIMUM_POPULATION:
            print("  not established, trying another seed", flush=True)
            continue

        brains = [a.brain for a in result["survivors"]]
        path = save_brains(brains)
        behaviour = summarise_population(brains[:40])
        print()
        print(f"saved {len(brains)} brains to {path}")
        print(f"  food seeking score: {behaviour['food_seeking']:.3f}  (chance is 0.25)")
        print(f"  reproduce rate change when crowded: {behaviour['reproduce_change']:+.3f}")
        return 0

    print(f"no attempt reached a population of {MINIMUM_POPULATION}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
