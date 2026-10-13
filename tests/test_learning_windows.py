"""Regressions for finite updates, inactive slots, and final learning windows."""
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from config import DEFAULT
from experiments.campaign_individual import run
from individual import Population
from rl import PolicyNet, run_enclosure, train


def one_step(pop, slot):
    dist = torch.distributions.Categorical(logits=pop.action_scores([slot], torch.ones(1, 12)))
    return {slot: ([dist.log_prob(torch.tensor([0]))[0]], [1.0])}


@pytest.mark.parametrize('mode', ['train', 'enclosure'])
def test_single_transition_is_finite_and_updates_policy(mode):
    cfg = DEFAULT.variant('single', n_ticks=1, n_initial_agents=1, n_food=0)
    initial = PolicyNet(seed=3)
    if mode == 'train':
        policy, history = train(cfg, False, iterations=1, seed=3, log_every=0)
    else:
        policy = PolicyNet(seed=3)
        opt = torch.optim.Adam(policy.parameters(), lr=.01)
        _, history = run_enclosure(cfg, policy, opt, seed=3, update_every=1,
                                   probe_every=0, verbose=False)
    assert math.isfinite(history[0]['loss'])
    assert all(torch.isfinite(p).all() for p in policy.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(initial.parameters(), policy.parameters()))


def test_inactive_slot_keeps_weights_and_adam_moments():
    pop = Population(2)
    a = pop.founder(PolicyNet(seed=1))
    b = pop.founder(PolicyNet(seed=2))
    opt = torch.optim.Adam(pop.parameters(), lr=.01)
    pop.learn(opt, one_step(pop, a))
    before = [p[a].detach().clone() for p in pop.parameters()]
    moments = [(opt.state[p]['exp_avg'][a].clone(), opt.state[p]['exp_avg_sq'][a].clone())
               for p in pop.parameters()]
    pop.learn(opt, one_step(pop, b))
    for p, value, (mean, square) in zip(pop.parameters(), before, moments):
        assert torch.equal(p[a], value)
        assert torch.equal(opt.state[p]['exp_avg'][a], mean)
        assert torch.equal(opt.state[p]['exp_avg_sq'][a], square)


@pytest.mark.parametrize('mode', ['enclosure', 'individual'])
@pytest.mark.parametrize('ticks,expected', [(3, 2), (4, 2)])
def test_final_window_updates_once_without_duplicate_flush(monkeypatch, mode, ticks, expected):
    cfg = DEFAULT.variant('final', n_ticks=ticks, n_initial_agents=2, n_food=0)
    policy = PolicyNet(seed=1)
    calls = []
    if mode == 'enclosure':
        opt = torch.optim.Adam(policy.parameters())
        original = opt.step

        def step(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(opt, 'step', step)
        run_enclosure(cfg, policy, opt, seed=1, update_every=2, probe_every=0, verbose=False)
    else:
        original = Population.learn

        def learn(self, opt, trajectories, gamma=.99):
            calls.append(sum(len(r) for _, r in trajectories.values()))
            return original(self, opt, trajectories, gamma)

        monkeypatch.setattr(Population, 'learn', learn)
        run(cfg, policy, seed=1, update_every=2, probe_every=0)
        assert calls == ([4, 2] if ticks == 3 else [4, 4])
    assert len(calls) == expected


def test_extinction_flushes_shared_terminal_experience(monkeypatch):
    cfg = DEFAULT.variant('extinct', n_ticks=5, n_initial_agents=1, n_food=0,
                          energy_start=.1, energy_cost_per_tick=1)
    policy = PolicyNet(seed=1)
    opt = torch.optim.Adam(policy.parameters())
    calls = []
    original = opt.step

    def step(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(opt, 'step', step)
    _, history = run_enclosure(cfg, policy, opt, seed=1, update_every=10,
                               probe_every=0, verbose=False)
    assert len(history) == 1 and history[0]['population'] == 0
    assert len(calls) == 1
    assert math.isfinite(history[0]['loss'])
