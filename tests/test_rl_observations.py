"""Policy inference must use the current tick's occupancy snapshot."""

import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import rl  # noqa: E402
from brain import ACTIONS, SENSE_INDEX  # noqa: E402
from config import DEFAULT  # noqa: E402
from experiments import campaign_individual  # noqa: E402
from individual import Population  # noqa: E402
from world import World  # noqa: E402


@pytest.mark.parametrize("loop", ["episode", "enclosure", "individual"])
def test_policy_sees_current_crowding_after_neighbour_moves(monkeypatch, loop):
    cfg = DEFAULT.variant(
        "observation_regression", width=10, height=10, wrap_edges=False,
        n_ticks=2, n_initial_agents=2, n_food=0, crowding_radius=1,
        crowding_cost_enabled=True, crowding_energy_cost=0.5,
    )
    world = World(10, 10, n_food=0, wrap_edges=False, rng=random.Random(0))
    positions = iter([(1, 1), (2, 1)])
    monkeypatch.setattr(world, "random_square", lambda: next(positions))
    monkeypatch.setattr(rl, "_build_world", lambda cfg, rng: world)
    monkeypatch.setattr(campaign_individual, "_build_world", lambda cfg, rng: world)

    # The first agent stays; its neighbour moves east out of range on tick one.
    monkeypatch.setattr(
        torch.distributions.Categorical, "sample",
        lambda self, sample_shape=torch.Size(): torch.tensor(
            [ACTIONS.index("stay"), ACTIONS.index("east")]
        ),
    )
    observed = []
    policy = rl.PolicyNet()
    if loop == "individual":
        original = Population.action_scores

        def capture(self, slots, senses):
            observed.append(senses.detach().clone())
            return original(self, slots, senses)

        monkeypatch.setattr(Population, "action_scores", capture)
        campaign_individual.run(cfg, policy, seed=0, update_every=10, probe_every=0)
    else:
        original = policy.forward

        def capture(senses):
            observed.append(senses.detach().clone())
            return original(senses)

        monkeypatch.setattr(policy, "forward", capture)
        if loop == "episode":
            rl.run_episode(cfg, policy, reward_offspring=False, seed=0)
        else:
            optimiser = torch.optim.Adam(policy.parameters())
            rl.run_enclosure(
                cfg, policy, optimiser, seed=0, update_every=10,
                probe_every=0, verbose=False,
            )

    assert len(observed) == 2
    crowding = SENSE_INDEX["crowding"]
    assert observed[0][:, crowding].tolist() == [1 / cfg.crowding_scale] * 2
    assert observed[1][:, crowding].tolist() == [0.0, 0.0]
    assert [(a.x, a.y) for a in world.agents] == [(1, 1), (4, 1)]
    # Costs still use tick-start occupancy, and inference still precedes costs.
    expected_energy = cfg.energy_start - 2 * cfg.energy_cost_per_tick - 0.5
    assert [a.energy for a in world.agents] == [expected_energy] * 2
    energy = SENSE_INDEX["energy"]
    assert observed[0][0, energy].item() == pytest.approx(cfg.energy_start / cfg.energy_max)
    assert observed[1][0, energy].item() == pytest.approx(
        (cfg.energy_start - cfg.energy_cost_per_tick - 0.5) / cfg.energy_max
    )
