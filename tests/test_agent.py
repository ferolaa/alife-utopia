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


def test_lifespan_is_infinite_when_ageing_is_off():
    from agent import draw_lifespan
    assert draw_lifespan(DEFAULT, random.Random(0)) == float("inf")


def test_lifespans_vary_between_individuals():
    from agent import draw_lifespan
    cfg = DEFAULT.variant("ageing", ageing_enabled=True)
    rng = random.Random(0)
    spans = [draw_lifespan(cfg, rng) for _ in range(200)]
    assert len(set(spans)) > 100                      # they are not all identical
    low = cfg.max_age * (1 - cfg.max_age_spread)
    high = cfg.max_age * (1 + cfg.max_age_spread)
    assert all(low <= s <= high for s in spans)


def test_reaching_your_lifespan_kills_you():
    cfg = DEFAULT.variant("ageing", ageing_enabled=True)
    world, agent, rng = make(cfg)
    agent.lifespan = 5
    agent.energy = 1000                               # plenty of energy, so only age can kill
    for _ in range(5):
        agent.act(world, cfg, rng)
    assert agent.alive is False


def test_the_young_cannot_breed():
    cfg = DEFAULT.variant("ageing", ageing_enabled=True)
    world, agent, rng = make(cfg)
    agent.energy = cfg.reproduce_threshold + 10
    agent.age = cfg.maturity_age - 1
    assert agent._try_reproduce(world, cfg, rng) is None


def test_the_old_cannot_breed():
    cfg = DEFAULT.variant("ageing", ageing_enabled=True)
    world, agent, rng = make(cfg)
    agent.energy = cfg.reproduce_threshold + 10
    agent.age = cfg.fertility_end_age + 1
    assert agent._try_reproduce(world, cfg, rng) is None


def test_breeding_works_inside_the_fertile_window():
    cfg = DEFAULT.variant("ageing", ageing_enabled=True)
    world, agent, rng = make(cfg)
    agent.energy = cfg.reproduce_threshold + 10
    agent.age = (cfg.maturity_age + cfg.fertility_end_age) // 2
    child = agent._try_reproduce(world, cfg, rng)
    assert child is not None
    assert child.lifespan != float("inf")             # the child ages too


def test_age_limits_are_ignored_when_ageing_is_off():
    cfg = DEFAULT                                      # ageing disabled
    world, agent, rng = make(cfg)
    agent.energy = cfg.reproduce_threshold + 10
    agent.age = 0                                      # far below maturity
    assert agent._try_reproduce(world, cfg, rng) is not None



NESTS = DEFAULT.variant("nests", nests_enabled=True, ageing_enabled=False)


def test_breeding_needs_a_nest_when_nests_are_on():
    world, agent, rng = make(NESTS)
    world.scatter_nests(5)
    agent.energy = NESTS.reproduce_threshold + 10
    # The agent is not standing on a nest, so it cannot breed however healthy it is.
    assert not world.is_free_nest(agent.x, agent.y)
    assert agent._try_reproduce(world, NESTS, rng) is None
    assert agent.energy == NESTS.reproduce_threshold + 10   # and it paid nothing


def test_breeding_on_a_nest_claims_it_for_the_child():
    world, agent, rng = make(NESTS)
    world.nests.add((agent.x, agent.y))
    agent.energy = NESTS.reproduce_threshold + 10
    child = agent._try_reproduce(world, NESTS, rng)
    assert child is not None
    assert child.nest == (agent.x, agent.y)
    assert (agent.x, agent.y) in world.occupied_nests
    assert child.x, child.y == (agent.x, agent.y)           # born in the nest


def test_a_nest_cannot_be_used_twice_at_once():
    world, agent, rng = make(NESTS)
    world.nests.add((agent.x, agent.y))
    agent.energy = 1000
    assert agent._try_reproduce(world, NESTS, rng) is not None
    # The nest is taken now, so a second attempt on the same square fails.
    assert agent._try_reproduce(world, NESTS, rng) is None


def test_growing_up_frees_the_nest():
    world, agent, rng = make(NESTS)
    square = (agent.x, agent.y)
    world.nests.add(square)
    agent.energy = 1000
    child = agent._try_reproduce(world, NESTS, rng)
    world.agents.append(child)
    assert square in world.occupied_nests
    child.age = NESTS.dependency_ticks          # old enough to leave the nest
    child.act(world, NESTS, rng)
    assert square not in world.occupied_nests
    assert child.nest is None


def test_nests_are_ignored_when_the_mechanism_is_off():
    cfg = DEFAULT                                # nests disabled
    world, agent, rng = make(cfg)
    agent.energy = cfg.reproduce_threshold + 10
    child = agent._try_reproduce(world, cfg, rng)
    assert child is not None
    assert child.nest is None



CARE = DEFAULT.variant("care", parental_care_enabled=True)


def _parent_and_pup(cfg=CARE):
    world, parent, rng = make(cfg)
    parent.energy = 1000
    pup = parent._try_reproduce(world, cfg, rng)
    world.agents.append(pup)
    world.rebuild_occupancy(cfg.crowding_radius)
    return world, parent, pup, rng


def test_a_newborn_starts_dependent_and_knows_its_parent():
    world, parent, pup, rng = _parent_and_pup()
    assert pup.is_dependent is True
    assert pup.parent is parent
    assert parent.dependents == [pup]


def test_a_pup_does_not_act_for_itself():
    world, parent, pup, rng = _parent_and_pup()
    before = (pup.x, pup.y, pup.energy)
    for _ in range(10):
        pup.act(world, CARE, rng)
    # It cannot move and it pays no energy. It is being fed.
    assert (pup.x, pup.y, pup.energy) == before


def test_an_attended_pup_stays_healthy():
    world, parent, pup, rng = _parent_and_pup()
    for _ in range(CARE.neglect_tolerance * 2):
        pup.act(world, CARE, rng)          # parent is right next to it
    assert pup.alive is True
    assert pup.neglect_ticks == 0


def test_an_abandoned_pup_dies_of_neglect():
    world, parent, pup, rng = _parent_and_pup()
    parent.x, parent.y = 0, 0              # far away
    pup.x, pup.y = 15, 15
    for _ in range(CARE.neglect_tolerance + 2):
        pup.act(world, CARE, rng)
    assert pup.alive is False
    assert pup.cause_of_death == "neglect"


def test_neglect_is_forgiven_when_the_parent_comes_back():
    world, parent, pup, rng = _parent_and_pup()
    parent.x, parent.y = 0, 0
    pup.x, pup.y = 15, 15
    for _ in range(CARE.neglect_tolerance - 1):
        pup.act(world, CARE, rng)
    assert pup.alive is True
    banked = pup.neglect_ticks
    parent.x, parent.y = pup.x, pup.y      # comes back
    for _ in range(banked):
        pup.act(world, CARE, rng)
    assert pup.neglect_ticks == 0          # fully recovered
    assert pup.alive is True


def test_a_dead_parent_cannot_care():
    world, parent, pup, rng = _parent_and_pup()
    parent.alive = False
    for _ in range(CARE.neglect_tolerance + 2):
        pup.act(world, CARE, rng)
    assert pup.alive is False
    assert pup.cause_of_death == "neglect"


def test_growing_up_ends_the_dependency():
    world, parent, pup, rng = _parent_and_pup()
    pup.age = CARE.dependency_ticks
    pup.energy = 50
    pup.act(world, CARE, rng)
    assert pup.is_dependent is False
    assert pup.parent is None
    assert parent.dependents == []


def test_causes_of_death_are_recorded():
    starving = DEFAULT.variant("starve", energy_cost_per_tick=1000.0)
    world, agent, rng = make(starving)
    agent.act(world, starving, rng)
    assert agent.cause_of_death == "starvation"

    old = DEFAULT.variant("old", ageing_enabled=True)
    world, agent, rng = make(old)
    agent.lifespan = 1
    agent.energy = 1000
    agent.act(world, old, rng)
    assert agent.cause_of_death == "old age"


def test_no_dependency_when_care_is_switched_off():
    world, parent, rng = make(DEFAULT)
    parent.energy = 1000
    pup = parent._try_reproduce(world, DEFAULT, rng)
    assert pup.is_dependent is False
    assert pup.parent is None
    assert parent.dependents == []


def test_a_parent_senses_where_its_pup_is():
    from brain import SENSE_INDEX
    world, parent, pup, rng = _parent_and_pup()
    pup.x, pup.y = parent.x + 2, parent.y
    senses = parent.sense(world, CARE)
    assert senses[SENSE_INDEX["pup_dx"]] > 0        # the pup is to the east
    assert senses[SENSE_INDEX["pup_need"]] > 0      # and it registers as having one


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
