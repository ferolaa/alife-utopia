"""Universe 25 feeder geometry must give every seed the same food capacity."""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from config import DEFAULT, UNIVERSE_25
from rl import _build_world
from world import World


def regions(w):
    s = w.feeder_spread
    return [{(fx+dx, fy+dy) for dx in range(-s, s+1) for dy in range(-s, s+1)}
            for fx, fy in sorted(w.feeders)]


def test_geometry_is_consistent_across_all_planned_and_guard_episode_seeds():
    for seed in (base*1000+i for base in range(5) for i in range(120)):
        w = _build_world(UNIVERSE_25, random.Random(seed))
        patches = regions(w)
        assert len(patches) == 4
        assert all(len(p) == 49 for p in patches)
        assert all(0 <= x < 48 and 0 <= y < 48 for p in patches for x, y in p)
        assert all(p.isdisjoint(q) for i, p in enumerate(patches) for q in patches[i+1:])
        assert len(w._food_squares()) == 196
        assert len(w.food) == 60 and w.food <= w._food_squares()


@pytest.mark.parametrize('seed', [0, 1, 2, 3082])
def test_matched_experiments_start_with_identical_geometry(seed):
    configurations = [
        UNIVERSE_25,
        UNIVERSE_25.variant('training', n_initial_agents=40, max_population=120, n_ticks=500),
        UNIVERSE_25.variant('individual'),
        UNIVERSE_25.variant('blocked', crowding_blocks_care=True),
        UNIVERSE_25.variant('intrusion', nest_intrusion_enabled=True),
        UNIVERSE_25.variant('damage', developmental_damage_enabled=True),
    ]
    worlds = [_build_world(cfg, random.Random(seed)) for cfg in configurations]
    assert all((w.feeders, w.food, w.nests) ==
               (worlds[0].feeders, worlds[0].food, worlds[0].nests) for w in worlds)


def test_placement_replays_despite_global_rng_activity_and_varies_with_seed():
    first = _build_world(UNIVERSE_25, random.Random(0))
    random.seed(89)
    random.random()
    second = _build_world(UNIVERSE_25, random.Random(0))
    assert (first.feeders, first.food, first.nests) == (second.feeders, second.food, second.nests)
    assert first.feeders != _build_world(UNIVERSE_25, random.Random(1)).feeders


@pytest.mark.parametrize('width,height,n', [(6, 48, 1), (14, 14, 5)])
def test_impossible_disjoint_placement_fails_before_partial_placement(width, height, n):
    w = World(width, height, 0, wrap_edges=False, rng=random.Random(0))
    with pytest.raises(ValueError, match='capacity'):
        w.place_feeders(n, disjoint_regions=True)
    assert w.feeders == set()


def test_ordinary_worlds_keep_unconstrained_placement():
    assert not DEFAULT.feeder_regions_disjoint
    w = World(6, 6, 0, wrap_edges=False, rng=random.Random(0))
    w.place_feeders(4)
    assert len(w.feeders) == 4
    assert len(w._food_squares()) < 196


def test_replacement_does_not_force_food_back_onto_eaten_square(monkeypatch):
    w = _build_world(UNIVERSE_25, random.Random(3082))
    eaten = sorted(w.food)[0]
    target = sorted(w._food_squares() - w.food)[0]
    feeder = next(f for f in sorted(w.feeders)
                  if abs(target[0]-f[0]) <= 3 and abs(target[1]-f[1]) <= 3)
    monkeypatch.setattr(w.rng, 'randrange', lambda n: sorted(w.feeders).index(feeder))
    offsets = iter([target[0]-feeder[0], target[1]-feeder[1]])
    monkeypatch.setattr(w.rng, 'randint', lambda a, b: next(offsets))
    assert w.take_food(*eaten)
    assert eaten not in w.food and target in w.food
    assert len(w.food) == 60 and len(w._food_squares()) == 196
