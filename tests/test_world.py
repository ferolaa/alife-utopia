"""Sanity checks for the grid world.

Run with:  python3 -m pytest tests/ -q      (or just: python3 tests/test_world.py)
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from world import World  # noqa: E402


def make_world(**kwargs):
    defaults = dict(width=10, height=10, n_food=5, rng=random.Random(0))
    defaults.update(kwargs)
    return World(**defaults)


def test_food_is_placed():
    w = make_world(n_food=7)
    assert len(w.food) == 7


def test_wrapping_moves_you_to_the_other_side():
    w = make_world(wrap_edges=True)
    assert w.normalise(10, 3) == (0, 3)
    assert w.normalise(-1, 3) == (9, 3)


def test_walls_clamp_instead_of_wrapping():
    w = make_world(wrap_edges=False)
    assert w.normalise(10, 3) == (9, 3)
    assert w.normalise(-1, 3) == (0, 3)


def test_eating_removes_food():
    w = make_world()
    square = next(iter(w.food))
    before = len(w.food)
    assert w.take_food(*square) is True
    assert len(w.food) == before - 1
    assert w.take_food(*square) is False   # nothing left to eat there


def test_unlimited_food_never_runs_out():
    w = make_world(n_food=5, food_unlimited=True)
    for _ in range(20):
        square = next(iter(w.food))
        w.take_food(*square)
    assert len(w.food) == 5


def test_sensing_points_towards_food():
    w = make_world(n_food=0)
    w.food.add((5, 3))
    dx, dy, closeness = w.nearest_food_direction(3, 3, vision=4)
    assert dx > 0          # food is to the right
    assert dy == 0         # and level with us
    assert 0 < closeness < 1


def test_sensing_returns_zeros_when_nothing_visible():
    w = make_world(n_food=0)
    assert w.nearest_food_direction(5, 5, vision=2) == (0.0, 0.0, 0.0)


class _Dummy:
    def __init__(self, x, y):
        self.x, self.y = x, y


def test_neighbour_counting():
    w = make_world()
    a, b, c = _Dummy(5, 5), _Dummy(5, 6), _Dummy(9, 9)
    w.agents = [a, b, c]
    w.rebuild_occupancy()
    assert w.count_neighbours(5, 5, radius=1, exclude=a) == 1   # only b is close
    assert w.count_neighbours(5, 5, radius=1) == 2              # a itself counts too


def test_nests_start_empty_and_can_be_claimed():
    w = make_world()
    assert w.nests == set()
    w.scatter_nests(3)
    assert len(w.nests) == 3
    square = next(iter(w.nests))
    assert w.is_free_nest(*square) is True
    assert w.claim_nest(*square) is True
    assert w.is_free_nest(*square) is False
    assert w.claim_nest(*square) is False      # cannot be claimed twice
    w.release_nest(*square)
    assert w.is_free_nest(*square) is True


def test_nest_sense_is_silent_when_there_are_no_nests():
    w = make_world()
    assert w.nearest_free_nest_direction(5, 5, vision=4) == (0.0, 0.0, 0.0)


def test_offset_takes_the_short_way_round_a_wrapped_grid():
    w = make_world(wrap_edges=True)            # 10 x 10
    assert w.offset(1, 1, 9, 9) == (-2, -2)    # not (8, 8)
    assert w.offset(9, 9, 1, 1) == (2, 2)


def test_offset_is_plain_subtraction_with_walls():
    w = make_world(wrap_edges=False)
    assert w.offset(1, 1, 9, 9) == (8, 8)


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
