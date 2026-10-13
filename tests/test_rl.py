"""Checks on the pen the creatures live in.

These exist because the pen used to be built in two different places, and the two copies
quietly stopped agreeing. Training happened in a world with food spread evenly and nests
anywhere, and the long run happened in a world with food piled around four feeders. The
nests were never put around the walls at all, even though the config asked for it, because
they had already been scattered elsewhere by the time anyone looked at the setting.

Nothing about that showed up as an error. The tests below would have caught it.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import DEFAULT, UNIVERSE_25  # noqa: E402
from rl import _build_world  # noqa: E402


def pen(seed=0, **changes):
    return _build_world(UNIVERSE_25.variant("test", **changes), random.Random(seed))


def in_band(world, x, y, band):
    return (x < band or x >= world.width - band
            or y < band or y >= world.height - band)


def test_the_pen_has_the_feeders_the_config_asks_for():
    world = pen()
    assert len(world.feeders) == UNIVERSE_25.n_feeders


def test_every_food_item_is_near_a_feeder():
    world = pen()
    spread = UNIVERSE_25.feeder_spread
    assert world.food
    for x, y in world.food:
        assert any(abs(x - fx) <= spread and abs(y - fy) <= spread
                   for fx, fy in world.feeders), f"food at {(x, y)} is not near any feeder"


def test_the_pen_has_the_nests_the_config_asks_for():
    world = pen()
    assert len(world.nests) == UNIVERSE_25.n_nests


def test_the_experiment_scatters_its_nests():
    """The pen used in the experiments deliberately does not put nests in the walls.

    That layout was tried and it kills the colony, for the reason written out in the config.
    This test is here so the setting cannot drift back without someone noticing.
    """
    assert not UNIVERSE_25.nests_on_perimeter


def test_asking_for_nests_in_the_walls_puts_them_there():
    """The setting has to actually work, even though the experiments do not use it.

    This is the bug that went unnoticed: nests were scattered before the setting was ever
    read, and scatter_nests stops once it has enough, so the call asking for walls did
    nothing. Every nest landing in the band is the whole point of the check.
    """
    band = UNIVERSE_25.perimeter_band
    world = pen(nests_on_perimeter=True)
    assert len(world.nests) == UNIVERSE_25.n_nests
    for x, y in world.nests:
        assert in_band(world, x, y, band), f"nest at {(x, y)} is in the middle of the pen"


def test_the_walls_layout_keeps_food_out_of_the_nesting_band():
    """With nests in the walls, feeding has to happen somewhere else or there is no point.

    Checked over many seeds, because the feeders are placed at random and one bad draw is
    all it would take.
    """
    band = UNIVERSE_25.perimeter_band
    for seed in range(50):
        world = pen(seed, nests_on_perimeter=True)
        assert not (world.nests & world.feeders)
        for x, y in world.food:
            assert not in_band(world, x, y, band), \
                f"seed {seed}: food at {(x, y)} is in the nesting band"


def test_the_population_cap_is_the_only_thing_that_changes_between_phases():
    """The training pen and the experiment pen must be the same pen.

    Same seed, same everything: the two worlds should come out identical square for square.
    Only the number of creatures allowed in it differs, and that is not part of the world.
    """
    training = pen(max_population=120, n_initial_agents=40)
    experiment = pen(max_population=3000)
    assert training.feeders == experiment.feeders
    assert training.food == experiment.food
    assert training.nests == experiment.nests


def test_a_world_without_feeders_still_scatters_food_everywhere():
    """The older uniform layout has to keep working, since it is what the comparison is."""
    world = _build_world(DEFAULT.variant("test"), random.Random(0))
    assert not world.feeders
    assert len(world.food) == DEFAULT.n_food
    assert not world.nests            # DEFAULT has no nests
    left = {(x, y) for x, y in world.food if x < world.width // 2}
    assert left and len(left) < len(world.food)   # spread over both halves


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
