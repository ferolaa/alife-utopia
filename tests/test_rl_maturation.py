"""First adult actions must come from RL and enter its learning trajectories."""

import math
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import rl  # noqa: E402
from agent import Agent  # noqa: E402
from brain import ACTIONS, Brain  # noqa: E402
from config import DEFAULT  # noqa: E402
from experiments import campaign_individual  # noqa: E402
from individual import Population  # noqa: E402
from world import World  # noqa: E402


@pytest.mark.parametrize("loop", ["episode", "enclosure", "individual"])
@pytest.mark.parametrize("ageing", [False, True])
def test_first_adult_action_uses_rl_and_is_recorded(monkeypatch, loop, ageing):
    cfg = DEFAULT.variant(
        "maturation", n_ticks=4, n_initial_agents=1, n_food=0,
        energy_start=50, parental_care_enabled=True, dependency_ticks=2,
        nests_enabled=True, crowding_cost_enabled=True, crowding_energy_cost=0.1,
        developmental_damage_enabled=True,
        ageing_enabled=ageing, maturity_age=1, fertility_end_age=2,
        max_age=100, max_age_spread=0,
    )
    world = World(10, 10, n_food=0, wrap_edges=False, rng=random.Random(0))
    world.nests = {(2, 2)}
    monkeypatch.setattr(world, "random_square", lambda: (2, 2))
    monkeypatch.setattr(rl, "_build_world", lambda cfg, rng: world)
    monkeypatch.setattr(campaign_individual, "_build_world", lambda cfg, rng: world)
    batches = []

    def sample(dist, sample_shape=torch.Size()):
        n = dist.logits.shape[0]
        batches.append(n)
        # One birth on tick one; parent then stays and the child moves east.
        actions = ["reproduce" if len(batches) == 1 else "stay"]
        actions += ["east"] * (n - 1)
        return torch.tensor([ACTIONS.index(a) for a in actions])

    monkeypatch.setattr(torch.distributions.Categorical, "sample", sample)
    numpy_calls = []
    monkeypatch.setattr(
        Brain, "decide", lambda *args, **kwargs: (numpy_calls.append(True) or "west")
    )
    adult_actions = []
    initial_ages = {}
    original_act = Agent.act

    def act(agent, world, cfg, rng, action=None):
        initial_ages.setdefault(agent, agent.age)
        if agent.parent is not None and agent.age + 1 >= agent.dependent_until:
            adult_actions.append(action)
        return original_act(agent, world, cfg, rng, action=action)

    monkeypatch.setattr(Agent, "act", act)
    policy = rl.PolicyNet()
    recorded = []
    if loop == "episode":
        log_probs, rewards, _ = rl.run_episode(cfg, policy, True, seed=0)
        recorded = [(len(log_probs[a]), rewards[a]) for a in world.agents]
    elif loop == "individual":
        original_learn = Population.learn

        def learn(pop, optimiser, trajectories, gamma=0.99):
            recorded.extend((len(steps), list(rewards)) for steps, rewards in trajectories.values())
            return original_learn(pop, optimiser, trajectories, gamma)

        monkeypatch.setattr(Population, "learn", learn)
        campaign_individual.run(cfg, policy, seed=0, update_every=4, probe_every=0)
    else:
        # The continuous shared loop stacks every recorded log probability at flush.
        original_stack = torch.stack

        def stack(tensors, *args, **kwargs):
            recorded.append(len(tensors))
            assert all(t.requires_grad for t in tensors)
            return original_stack(tensors, *args, **kwargs)

        monkeypatch.setattr(torch, "stack", stack)
        optimiser = torch.optim.Adam(policy.parameters())
        _, history = rl.run_enclosure(
            cfg, policy, optimiser, seed=0, update_every=4,
            probe_every=0, verbose=False,
        )
        assert math.isfinite(history[-1]["loss"])

    assert numpy_calls == [], "a newly independent pup consulted its NumPy brain"
    assert adult_actions == ["east"]
    assert batches == [1, 1, 2, 2]
    parent, child = world.agents
    assert (child.x, child.y) == (4, 2)
    assert parent.age == initial_ages[parent] + 4
    assert child.age == 3
    assert child.lifespan == (100 if ageing else float("inf"))
    assert child.alive and not child.is_dependent
    assert child.parent is None and parent.dependents == []
    assert child.nest is None and world.occupied_nests == set()
    assert child.neglect_suffered == 0 and child.impairment == 0
    assert child.energy == pytest.approx(cfg.reproduce_cost - 2 * (cfg.energy_cost_per_tick + 0.1))
    if loop == "enclosure":
        assert recorded == [6]  # four parent choices and both adult child choices
    else:
        assert recorded == [
            (4, [0.01, 0.01, 1.01, 0.01]),
            (2, [0.01, 0.01]),
        ]
