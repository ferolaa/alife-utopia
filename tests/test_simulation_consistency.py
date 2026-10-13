"""Regressions for bounded food placement and spatial measurements."""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from agent import Agent
from brain import Brain
from config import DEFAULT, UNIVERSE_25
from rl import _build_world
from world import World


def world(**changes):
    args = dict(width=10, height=10, n_food=0, wrap_edges=False, rng=random.Random(0))
    args.update(changes)
    return World(**args)


class Body:
    def __init__(self, x, y):
        self.x, self.y = x, y


def test_impossible_initial_food_request_fails_without_partial_placement():
    w = world()
    w.feeders = {(5, 5)}
    with pytest.raises(ValueError, match='capacity'):
        w.scatter_food(10, spread=1)
    assert w.food == set()


def test_original_infeasible_food_request_is_reported_not_silently_adjusted():
    cfg = UNIVERSE_25.variant('original_infeasible_budget', n_food=260)
    with pytest.raises(ValueError, match='260.*capacity'):
        _build_world(cfg, random.Random(0))
    assert cfg.n_food == 260


@pytest.mark.parametrize('wrap,feeders,spread,capacity', [
    (False, {(0, 0)}, 1, 4),
    (True, {(0, 0)}, 1, 9),
    (False, {(4, 4), (5, 4)}, 1, 12),
    (True, {(0, 0)}, 10, 100),
])
def test_capacity_accounts_for_walls_overlap_and_wrapping(wrap, feeders, spread, capacity):
    w = world(wrap_edges=wrap)
    w.feeders = feeders
    w.scatter_food(capacity, spread=spread)
    assert len(w.food) == capacity
    before = w.food.copy()
    with pytest.raises(ValueError, match='capacity'):
        w.scatter_food(1, spread=spread)
    assert w.food == before


def test_feasible_placement_finishes_even_when_random_candidates_repeat(monkeypatch):
    w = world()
    monkeypatch.setattr(w, 'random_square', lambda: (0, 0))
    w.scatter_food(3)
    assert len(w.food) == 3


@pytest.mark.parametrize('spread', [0, 1, 4])
@pytest.mark.parametrize('unlimited', [False, True])
def test_respawning_uses_configured_spread(spread, unlimited):
    w = world(feeder_spread=spread, food_unlimited=unlimited, food_respawn_rate=2)
    w.feeders = {(5, 5)}
    w.scatter_food(1)
    for _ in range(15):
        square = next(iter(w.food))
        assert w.take_food(*square)
        w.respawn_step()
        assert w.food
        assert all(abs(x-5) <= spread and abs(y-5) <= spread for x, y in w.food)
        if unlimited:
            assert len(w.food) == 1


def test_limited_respawning_saturates_without_exceeding_capacity():
    w = world(feeder_spread=0, food_respawn_rate=4.5)
    w.feeders = {(5, 5)}
    for _ in range(10):
        w.respawn_step()
    assert w.food == {(5, 5)}


@pytest.mark.parametrize('cached_radius,requested_radius', [(1, 2), (2, 1), (1, 0)])
@pytest.mark.parametrize('wrap', [False, True])
def test_neighbour_cache_respects_query_radius(cached_radius, requested_radius, wrap):
    w = world(wrap_edges=wrap)
    a = Body(1, 1)
    w.agents = [a, Body(3, 1), Body(1, 2), Body(9, 1)]
    w.rebuild_occupancy()
    expected = w.count_neighbours(1, 1, requested_radius, exclude=a)
    w.rebuild_occupancy(cached_radius)
    assert w.count_neighbours(1, 1, requested_radius, exclude=a) == expected


def test_wall_clustering_uses_clipped_neighbourhoods():
    w = world()
    w.agents = [Body(0, 0), Body(0, 0), Body(0, 0)]
    w.rebuild_occupancy(1)
    counts = [w.count_neighbours(a.x, a.y, 1, exclude=a) for a in w.agents]
    assert w.clustering(counts, 1) == pytest.approx(25)


def test_uniform_null_clustering_is_one_at_edges_and_interior():
    w = world()
    w.agents = [Body(0, 0), Body(0, 5), Body(5, 5)]
    # The expected counts conditional on each observed position, for IID uniform peers.
    expected = [2 * area / 100 for area in (4, 6, 9)]
    assert w.clustering(expected, 1) == pytest.approx(1)


def test_eating_after_costs_can_rescue_an_agent_before_death_check():
    cfg = DEFAULT.variant('rescue', energy_cost_per_tick=2, energy_from_food=3)
    w = world()
    a = Agent(5, 5, 1, Brain.random(random.Random(0)))
    w.agents = [a]
    w.food = {(5, 5)}
    w.rebuild_occupancy(cfg.crowding_radius)
    a.act(w, cfg, random.Random(0), action='stay')
    assert a.alive and a.energy == 2


def test_world_builder_passes_spread_to_unlimited_respawning():
    cfg = DEFAULT.variant('spread_zero', n_feeders=1, feeder_spread=0,
                          n_food=1, food_unlimited=True)
    w = _build_world(cfg, random.Random(0))
    for _ in range(10):
        assert w.food == w.feeders
        w.take_food(*next(iter(w.food)))


@pytest.mark.parametrize('wrap', [False, True])
def test_large_radius_counts_each_agent_only_once(wrap):
    w = world(width=3, height=2, wrap_edges=wrap)
    a = Body(0, 0)
    w.agents = [a, Body(2, 1)]
    for cached in (None, 4):
        w.rebuild_occupancy(cached)
        assert w.count_neighbours(0, 0, 4, exclude=a) == 1
        assert w.clustering([1, 1], 4) == pytest.approx(1)


@pytest.mark.parametrize('mode', ['enclosure', 'individual'])
def test_campaign_clustering_pairs_counts_with_tick_start_positions(monkeypatch, mode):
    sys.path.insert(0, str(ROOT))
    import torch
    import rl
    from brain import ACTIONS
    from experiments import campaign_individual
    cfg = DEFAULT.variant('snapshot', n_ticks=1, n_initial_agents=2, n_food=0,
                          crowding_radius=1, wrap_edges=False)
    w = world()
    points = iter([(0, 0), (0, 0)])
    monkeypatch.setattr(w, 'random_square', lambda: next(points))
    monkeypatch.setattr(rl, '_build_world', lambda cfg, rng: w)
    monkeypatch.setattr(campaign_individual, '_build_world', lambda cfg, rng: w)
    monkeypatch.setattr(torch.distributions.Categorical, 'sample',
                        lambda self, sample_shape=torch.Size():
                        torch.tensor([ACTIONS.index('east'), ACTIONS.index('east')]))
    p = rl.PolicyNet(seed=0)
    if mode == 'enclosure':
        _, history = rl.run_enclosure(cfg, p, torch.optim.Adam(p.parameters()),
                                     probe_every=0, verbose=False)
    else:
        _, _, history = campaign_individual.run(cfg, p, seed=0, probe_every=0)
    assert [(a.x, a.y) for a in w.agents] == [(1, 0), (1, 0)]
    assert history[0]['mean_neighbours'] == 1
    assert history[0]['clustering'] == pytest.approx(25)  # corner patch, not post-move patch


@pytest.mark.parametrize('training_seed,iterations', [(0, 120), (1, 120), (2, 120), (3, 100), (4, 100)])
def test_actual_preset_food_budget_fits_all_planned_episode_seeds(training_seed, iterations):
    # This also includes campaign seeds 0–2 and representation seeds 100–105.
    for iteration in range(iterations):
        seed = training_seed * 1000 + iteration
        w = _build_world(UNIVERSE_25, random.Random(seed))
        assert len(w.food) == UNIVERSE_25.n_food == 60
        assert w.food <= w._food_squares()
        assert len(w.feeders) == UNIVERSE_25.n_feeders == 4
        assert w.feeder_spread == UNIVERSE_25.feeder_spread == 3


def test_previous_limiting_seed_has_fixed_capacity_and_stays_replenished():
    w = _build_world(UNIVERSE_25, random.Random(3082))
    assert len(w._food_squares()) == 196
    assert UNIVERSE_25.n_food == 60
    for _ in range(100):
        assert w.take_food(*next(iter(w.food)))
        assert len(w.food) == 60
        assert w.food <= w._food_squares()
