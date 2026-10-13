"""Seeded simulations and training must be independent of ambient RNG state."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from agent import Agent
from config import DEFAULT
from experiments.campaign_individual import run
from represent import probe
from rl import PolicyNet, run_episode, run_enclosure, train

CFG = DEFAULT.variant('replay', width=12, height=12, n_food=20,
                      n_initial_agents=4, max_population=16, n_ticks=20)


def perturb():
    random.seed(987)
    np.random.seed(987)
    torch.manual_seed(987)
    random.random()
    np.random.random(50)
    torch.rand(50)


def weights(module):
    return [p.detach().clone() for p in module.parameters()]


@pytest.mark.parametrize('mode', ['episode', 'enclosure', 'individual'])
def test_seeded_simulation_replays_actions_histories_and_learning(monkeypatch, mode):
    actions = []
    original = Agent.act

    def act(self, world, cfg, rng, action=None):
        actions.append((self.x, self.y, self.age, action))
        return original(self, world, cfg, rng, action=action)

    monkeypatch.setattr(Agent, 'act', act)

    def execute(seed):
        actions.clear()
        policy = PolicyNet(seed=12)
        if mode == 'episode':
            logs, rewards, history = run_episode(CFG, policy, True, seed)
            learned = [torch.stack(v).detach().clone() for v in logs.values()]
            history = (history, list(rewards.values()))
        elif mode == 'enclosure':
            opt = torch.optim.Adam(policy.parameters(), lr=.003)
            _, history = run_enclosure(CFG, policy, opt, seed=seed,
                                       update_every=7, probe_every=0, verbose=False)
            learned = weights(policy)
        else:
            _, pop, history = run(CFG, policy, seed, update_every=7, probe_every=0)
            learned = weights(pop)
        return list(actions), history, learned

    first = execute(3)
    perturb()
    rng_before = torch.get_rng_state().clone()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    second = execute(3)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert first[:2] == second[:2]
    assert all(torch.equal(a, b) for a, b in zip(first[2], second[2]))
    assert first[0] != execute(4)[0]


def test_seeded_initialization_training_and_evaluation_are_isolated():
    first, history = train(CFG, True, iterations=3, seed=5, log_every=0)
    evaluated = probe(first.to_brain(), trials=50, seed=8)
    perturb()
    rng_before = torch.get_rng_state().clone()
    second, repeated = train(CFG, True, iterations=3, seed=5, log_every=0)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert history == repeated
    assert all(torch.equal(a, b) for a, b in zip(weights(first), weights(second)))
    assert evaluated == probe(second.to_brain(), trials=50, seed=8)
    assert any(not torch.equal(a, b) for a, b in
               zip(weights(PolicyNet(seed=5)), weights(PolicyNet(seed=6))))
