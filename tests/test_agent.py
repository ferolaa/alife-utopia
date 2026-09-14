"""Sanity checks for a single creature's energy economy and behaviour."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import Agent  # noqa: E402
from brain import Brain  # noqa: E402
from config import DEFAULT  # noqa: E402
from world import World  # noqa: E402


def make(cfg=DEFAULT, n_food=0, seed=0):
    rng = random.Random(seed)
    world = World(
        width=cfg.width, height=cfg.height, n_food=n_food,
        food_unlimited=cfg.food_unlimited, wrap_edges=cfg.wrap_edges, rng=rng,
    )
    agent = Agent(10, 10, cfg.energy_start, Brain.random(rng))
    world.agents = [agent]
    world.rebuild_occupancy()
    return world, agent, rng


def test_sense_has_one_value_per_input():
    world, agent, _ = make()
    assert len(agent.sense(world, DEFAULT)) == Brain.N_INPUTS


def test_sense_values_stay_in_range():
    from brain import SENSE_INDEX
    world, agent, _ = make(n_food=50)
    senses = agent.sense(world, DEFAULT)
    # Every input must arrive on the scale the network expects. A sense that drifts
    # outside this range would quietly dominate the first layer.
    for value in senses:
        assert -1.0 <= value <= 1.0
    for name in ("food_closeness", "energy", "age", "crowding", "nest_closeness", "pup_need"):
        assert 0.0 <= senses[SENSE_INDEX[name]] <= 1.0


def test_senses_are_zero_for_disabled_mechanisms():
    from brain import SENSE_INDEX
    world, agent, _ = make()
    senses = agent.sense(world, DEFAULT)
    # No nests exist and no pups exist, so those senses must report nothing.
    for name in ("nest_dx", "nest_dy", "nest_closeness", "pup_dx", "pup_dy", "pup_need"):
        assert senses[SENSE_INDEX[name]] == 0.0


def test_living_costs_energy():
    world, agent, rng = make()
    before = agent.energy
    agent.act(world, DEFAULT, rng)
    assert agent.energy < before


def test_landing_on_food_feeds_you():
    # The creature acts first and eats wherever it ends up, so food is placed on its
    # square and every square it could step to - otherwise this test would depend on
    # which way a random brain happens to move.
    cfg = DEFAULT.variant("no-move-cost", energy_cost_per_tick=0.0)
    world, agent, rng = make(cfg)
    for dx, dy in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
        world.food.add(world.normalise(agent.x + dx, agent.y + dy))
    before = agent.energy
    agent.act(world, cfg, rng)
    assert agent.energy > before
    assert not world.has_food(agent.x, agent.y)   # the food it landed on is gone


def test_energy_is_capped():
    cfg = DEFAULT.variant("rich", energy_cost_per_tick=0.0, energy_from_food=1000.0)
    world, agent, rng = make(cfg)
    world.food.add((agent.x, agent.y))
    agent.act(world, cfg, rng)
    assert agent.energy <= cfg.energy_max


def test_running_out_of_energy_kills():
    cfg = DEFAULT.variant("harsh", energy_cost_per_tick=100.0)
    world, agent, rng = make(cfg)
    agent.act(world, cfg, rng)
    assert agent.alive is False


def test_reproduction_fails_below_the_threshold():
    cfg = DEFAULT
    world, agent, rng = make(cfg)
    agent.energy = cfg.reproduce_threshold - 1
    assert agent._try_reproduce(world, cfg, rng) is None
    assert agent.children == 0


def test_reproduction_costs_the_parent_and_pays_the_child():
    cfg = DEFAULT
    world, agent, rng = make(cfg)
    agent.energy = cfg.reproduce_threshold + 5
    before = agent.energy
    child = agent._try_reproduce(world, cfg, rng)
    assert child is not None
    assert agent.energy == before - cfg.reproduce_cost
    assert child.energy == cfg.reproduce_cost
    assert agent.children == 1


def test_a_child_brain_is_a_mutated_copy_of_the_parent():
    cfg = DEFAULT
    world, agent, rng = make(cfg)
    agent.energy = cfg.reproduce_threshold + 5
    child = agent._try_reproduce(world, cfg, rng)
    diff = abs(child.brain.weights - agent.brain.weights)
    assert diff.mean() > 0        # it mutated
    assert diff.mean() < 1.0      # but it is still its parent's brain


def test_crowding_costs_energy_only_when_enabled():
    crowded = DEFAULT.variant("crowded", crowding_cost_enabled=True)
    calm = DEFAULT.variant("calm", crowding_cost_enabled=False)

    def energy_after_one_tick(cfg):
        rng = random.Random(0)
        world = World(width=20, height=20, n_food=0, rng=rng)
        agent = Agent(10, 10, 50.0, Brain.random(random.Random(1)))
        crowd = [Agent(10, 10, 50.0, Brain.random(random.Random(2))) for _ in range(5)]
        world.agents = [agent] + crowd
        world.rebuild_occupancy()
        agent.act(world, cfg, rng)
        return agent.energy

    assert energy_after_one_tick(crowded) < energy_after_one_tick(calm)


def test_age_increases_each_tick():
    world, agent, rng = make()
    for _ in range(5):
        agent.act(world, DEFAULT, rng)
    assert agent.age == 5


def test_neighbours_are_counted_once_and_remembered():
    world, agent, rng = make()
    assert agent.neighbours is None        # has not acted yet
    agent.act(world, DEFAULT, rng)
    assert agent.neighbours is not None    # the count is kept for reuse


def test_refunding_a_birth_restores_the_parent():
    cfg = DEFAULT
    world, agent, rng = make(cfg)
    agent.energy = cfg.reproduce_threshold + 5
    before = agent.energy
    child = agent._try_reproduce(world, cfg, rng)
    assert child is not None
    agent.refund_birth(cfg)
    assert agent.energy == before          # energy given back
    assert agent.children == 0             # and it does not count as a child


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
