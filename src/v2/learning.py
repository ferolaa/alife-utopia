"""On-policy full-episode Monte Carlo REINFORCE, optionally with a learned baseline.

No PPO, no reused rollout epochs, no batch return standardization. A separate
critic predicts MC returns from pre-action observations. Actor advantages are
computed BEFORE this batch trains the critic and are detached.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch.distributions import Categorical

from .data import event_returns
from .models import Policy, Value


@dataclass(frozen=True)
class LearnerConfig:
    algorithm: str = 'reinforce'
    architecture: str = 'original'
    seed: int = 0
    gamma: float = .99
    actor_lr: float = .001
    value_lr: float = .001
    entropy_coefficient: float = 0.
    max_grad_norm: float | None = None
    reward_scope: str = 'colony'

    def __post_init__(self):
        if self.algorithm not in ('reinforce', 'reinforce_value'):
            raise ValueError('supported algorithms: reinforce, reinforce_value; PPO is not implemented')
        if self.reward_scope not in ('colony', 'individual_surrogate'):
            raise ValueError('invalid reward scope')
        if not 0 <= self.gamma <= 1:
            raise ValueError('invalid discount')
        if not all(math.isfinite(v) and v > 0 for v in (self.actor_lr, self.value_lr)):
            raise ValueError('learning rates must be finite and positive')
        if not math.isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0:
            raise ValueError('invalid entropy coefficient')
        if self.max_grad_norm is not None and (not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError('invalid gradient limit')


def discounted_returns(rewards, gamma, *, terminated=False, truncated=False, bootstrap=None):
    """Small independent reference for boundary tests and future actor-critic work."""
    if terminated and truncated:
        raise ValueError('a boundary cannot be both terminal and truncated')
    if not 0 <= gamma <= 1:
        raise ValueError('invalid discount')
    if not terminated and bootstrap is None:
        raise ValueError('nonterminal/truncated paths require explicit bootstrap')
    running = 0. if terminated else float(bootstrap)
    returns = []
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    return list(reversed(returns))


def policy_loss(log_probabilities, advantages, discounts, episodes=1):
    """Sum score terms per world; avoid a policy-dependent decision-count divisor."""
    return -(log_probabilities * advantages.detach() * discounts).sum() / episodes


class Learner:
    def __init__(self, config=LearnerConfig()):
        self.config = config
        self.policy = Policy(config.architecture, config.seed)
        self.value = Value(config.seed + 1) if config.algorithm == 'reinforce_value' else None
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.actor_lr)
        self.value_optimizer = (torch.optim.Adam(self.value.parameters(), lr=config.value_lr)
                                if self.value is not None else None)
        self.updates = 0

    def update(self, episodes):
        if not episodes:
            raise ValueError('at least one episode required')
        if any(e.consumed for e in episodes):
            raise ValueError('rollout has already been consumed')
        records, targets = [], []
        for episode in episodes:
            targets.extend(event_returns(episode, self.config.gamma, self.config.reward_scope))
            records.extend(episode.transitions)
        if not records:
            for episode in episodes:
                episode.consumed = True
            return dict(optimizer_updates=self.updates, actor_loss=0., value_loss=None,
                        entropy=None, gradient_norm=0., value_gradient_norm=None)
        observations = torch.tensor([t.observation for t in records], dtype=torch.float32)
        actions = torch.tensor([t.action for t in records])
        returns = torch.tensor(targets, dtype=torch.float32)
        discounts = torch.tensor([self.config.gamma ** t.tick for t in records])
        distribution = Categorical(logits=self.policy(observations))
        # A rollout is single-use and must come from these exact actor weights.
        old = torch.tensor([t.log_probability for t in records])
        if not torch.allclose(distribution.log_prob(actions).detach(), old, atol=2e-5, rtol=1e-5):
            raise ValueError('stale or foreign on-policy rollout')
        baseline = self.value(observations) if self.value is not None else torch.zeros_like(returns)
        advantages = returns - baseline.detach()
        entropy = distribution.entropy()
        loss = policy_loss(distribution.log_prob(actions), advantages, discounts, len(episodes))
        # Optional exploration surrogate, disabled by default and logged in metadata.
        loss = loss - self.config.entropy_coefficient * entropy.mean()
        self.optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
            self.config.max_grad_norm if self.config.max_grad_norm is not None else float('inf'),
            error_if_nonfinite=True)
        self.optimizer.step()
        value_loss = value_norm = None
        if self.value is not None:
            objective = (baseline - returns).square().mean()
            self.value_optimizer.zero_grad()
            objective.backward()
            value_norm = float(torch.nn.utils.clip_grad_norm_(self.value.parameters(), float('inf'),
                                                            error_if_nonfinite=True))
            self.value_optimizer.step()
            value_loss = float(objective.detach())
        for episode in episodes:
            episode.consumed = True
        self.updates += 1
        return dict(optimizer_updates=self.updates, actor_loss=float(loss.detach()),
                    value_loss=value_loss, entropy=float(entropy.mean().detach()),
                    gradient_norm=float(norm), value_gradient_norm=value_norm,
                    return_mean=float(returns.mean()), advantage_mean=float(advantages.mean()),
                    advantage_std=float(advantages.std(unbiased=False)))

    def state(self):
        return dict(config=asdict(self.config), actor=self.policy.state_dict(),
                    optimizer=self.optimizer.state_dict(), updates=self.updates,
                    value=self.value.state_dict() if self.value is not None else None,
                    value_optimizer=self.value_optimizer.state_dict() if self.value is not None else None)

    @classmethod
    def restore(cls, state):
        learner = cls(LearnerConfig(**state['config']))
        learner.policy.load_state_dict(state['actor'])
        learner.optimizer.load_state_dict(state['optimizer'])
        learner.updates = state['updates']
        if learner.value is not None:
            learner.value.load_state_dict(state['value'])
            learner.value_optimizer.load_state_dict(state['value_optimizer'])
        return learner
