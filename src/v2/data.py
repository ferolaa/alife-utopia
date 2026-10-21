"""Clocked records, including rewards paid after the recipient's death."""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class RewardConfig:
    objective: str = 'survival'
    survival: float = .01
    independence: float = 1.

    def __post_init__(self):
        if self.objective not in ('survival', 'offspring'):
            raise ValueError('objective must be survival or offspring')
        if not all(math.isfinite(v) and v >= 0 for v in (self.survival, self.independence)):
            raise ValueError('reward coefficients must be finite and nonnegative')


@dataclass(frozen=True)
class Event:
    tick: int
    recipient: int
    amount: float
    kind: str
    subject: int | None = None


@dataclass
class Transition:
    tick: int
    agent: int
    observation: tuple
    action: int
    log_probability: float
    terminated: bool = False
    truncated: bool = False


@dataclass
class Episode:
    world_seed: int
    action_seed: int
    initial_hash: str
    horizon: int
    transitions: list[Transition] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    truncated: bool = False
    consumed: bool = False

    @property
    def decisions(self):
        return len(self.transitions)


def event_returns(episode, gamma, scope='colony'):
    """Return-to-go at actual action times, not compressed agent-list positions.

    colony: all reward events, including events after an actor dies. Together with
    gamma**tick weighting this estimates gradient of E[sum gamma**t R_colony(t)].
    individual_surrogate: own rewards only; omits cross-agent causal score terms.
    Useful for auditing legacy credit, NOT the gradient of colony total reward.
    """
    if not 0 <= gamma <= 1 or scope not in ('colony', 'individual_surrogate'):
        raise ValueError('invalid discount or reward scope')
    if episode.truncated:
        raise ValueError('MC updates require completed finite-horizon episodes; no silent zero bootstrap')
    # Linear in events + decisions for colony returns; one sparse clock per
    # recipient for the individual surrogate. Avoid quadratic event scans.
    streams = {}
    for e in episode.events:
        key = 0 if scope == 'colony' else e.recipient
        streams.setdefault(key, {})[e.tick] = streams.setdefault(key, {}).get(e.tick, 0.) + e.amount
    clocks = {}
    last_tick = max([episode.horizon - 1, *[e.tick for e in episode.events]])
    for key, events in streams.items():
        running, clock = 0., {}
        for tick in range(last_tick, -1, -1):
            running = events.get(tick, 0.) + gamma * running
            clock[tick] = running
        clocks[key] = clock
    return [clocks.get(0 if scope == 'colony' else t.agent, {}).get(t.tick, 0.)
            for t in episode.transitions]
