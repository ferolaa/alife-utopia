"""Saving and loading populations of evolved brains.

Every experiment in this project starts from the same population of creatures that already
know how to survive and raise young. That population is produced once, by a founding run,
and then reused.

The reason is that the question being asked is Calhoun's question: does an established
population break down under pressure. It is not a question about whether such a population
can arise in the first place. Starting every run from random brains would answer the second
question loudly enough to drown out the first, because a population of random brains spends
thousands of ticks nearly dying while it discovers parental care, and whether it survives
that is mostly luck. Calhoun did not begin with mice that had to invent mothering. Neither
do we.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from brain import Brain

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FOUNDERS_FILE = "founders.npz"


def save_brains(brains, path: Path | None = None) -> Path:
    """Write a population's weights to a single compressed file."""
    target = Path(path or RESULTS_DIR / FOUNDERS_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.stack([b.weights for b in brains]) if brains else np.zeros((0, Brain.N_WEIGHTS))
    np.savez_compressed(target, weights=stacked)
    return target


def load_brains(path: Path | None = None) -> list[Brain]:
    """Read a saved population back.

    Refuses to load weights of the wrong size rather than failing later in a confusing way.
    The brain's input layer has already changed once during this project, and a saved
    population from before that change is not usable.
    """
    source = Path(path or RESULTS_DIR / FOUNDERS_FILE)
    if not source.exists():
        raise FileNotFoundError(
            f"no saved population at {source}. Run experiments/found_population.py first."
        )
    stacked = np.load(source)["weights"]
    if stacked.shape[1:] != (Brain.N_WEIGHTS,):
        raise ValueError(
            f"saved brains have {stacked.shape[1:]} weights but this version of the brain "
            f"expects {Brain.N_WEIGHTS}. The saved population is out of date."
        )
    return [Brain(row.copy()) for row in stacked]


def draw_founders(brains: list[Brain], n: int, rng: random.Random) -> list[Brain]:
    """Pick n brains to start a run with.

    Sampling with replacement, so a run can be larger than the saved population, and so
    that different seeds start from different draws. That gives runs genuine variation
    without making survival a coin flip, which is the whole point of founding them from
    competent stock in the first place.
    """
    if not brains:
        raise ValueError("cannot draw founders from an empty population")
    return [Brain(rng.choice(brains).weights.copy()) for _ in range(n)]
