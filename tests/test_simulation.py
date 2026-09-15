"""Checks on the simulation loop and the numbers it records."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import simulation  # noqa: E402
from agent import Agent  # noqa: E402
from brain import Brain  # noqa: E402
from config import DEFAULT  # noqa: E402
from world import World  # noqa: E402


def _world_with(positions, width=40, height=40):
    w = World(width=width, height=height, n_food=0, rng=random.Random(0))
    w.agents = [Agent(x, y, 50.0, Brain.random(random.Random(i)))
                for i, (x, y) in enumerate(positions)]
    w.rebuild_occupancy(DEFAULT.crowding_radius)
    for a in w.agents:
        a.neighbours = w.count_neighbours(a.x, a.y, DEFAULT.crowding_radius, exclude=a)
    return w


def test_clustering_is_about_one_when_agents_are_spread_out():
    """Evenly spread agents should score near one, whatever the population size.

    This is the check that would have caught the original formula, which used
    population * patch / cells - 1 instead of (population - 1) * patch / cells. The two
    barely differ for a dense population but the wrong one collapses towards zero for a
    sparse one, sending the ratio to absurd values exactly when the pen is nearly empty.
    """
    rng = random.Random(1)
    # Sixty agents scattered at random over a large grid: no clustering by construction.
    positions = [(rng.randrange(40), rng.randrange(40)) for _ in range(60)]
    w = _world_with(positions)
    measured = [a.neighbours for a in w.agents]
    score = simulation._clustering(w, DEFAULT, measured)
    assert 0.3 < score < 3.0, score


def test_a_sparse_population_does_not_produce_an_absurd_score():
    # The regime where the old formula exploded: few agents in a big pen.
    rng = random.Random(2)
    positions = [(rng.randrange(40), rng.randrange(40)) for _ in range(12)]
    w = _world_with(positions)
    measured = [a.neighbours for a in w.agents]
    score = simulation._clustering(w, DEFAULT, measured)
    assert score < 10.0, f"sparse population scored {score}, which is the old bug"


def test_clustering_rises_when_agents_pile_up():
    # Everyone on one square: as clustered as it is possible to be.
    w = _world_with([(20, 20)] * 60)
    measured = [a.neighbours for a in w.agents]
    piled = simulation._clustering(w, DEFAULT, measured)

    rng = random.Random(3)
    spread = _world_with([(rng.randrange(40), rng.randrange(40)) for _ in range(60)])
    spread_score = simulation._clustering(spread, DEFAULT, [a.neighbours for a in spread.agents])

    assert piled > spread_score * 5


def test_clustering_is_zero_for_an_empty_or_single_population():
    assert simulation._clustering(_world_with([]), DEFAULT, []) == 0.0
    w = _world_with([(5, 5)])
    assert simulation._clustering(w, DEFAULT, [0]) == 0.0


def test_a_short_run_records_every_tick():
    cfg = DEFAULT.variant("short", n_ticks=25, n_initial_agents=10)
    result = simulation.run(cfg)
    assert len(result["history"]) == 25
    assert all("clustering" in row for row in result["history"])


def test_runs_are_reproducible_from_the_seed():
    cfg = DEFAULT.variant("repeat", n_ticks=60, n_initial_agents=20, seed=7)
    first = [r["population"] for r in simulation.run(cfg)["history"]]
    second = [r["population"] for r in simulation.run(cfg)["history"]]
    assert first == second


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
                print(f"PASS {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
