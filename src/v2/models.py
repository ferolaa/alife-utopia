"""CPU policies with isolated initialization and a common categorical interface."""
import torch
from torch import nn

ARCHITECTURES = {'linear': (), 'original': (12,), 'deep': (64, 64)}


def network(hidden, outputs):
    layers, size = [], 12
    for width in hidden:
        layers.extend([nn.Linear(size, width), nn.Tanh()])
        size = width
    layers.append(nn.Linear(size, outputs))
    return nn.Sequential(*layers)


class Policy(nn.Module):
    def __init__(self, architecture='original', seed=0):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f'unknown architecture: {architecture}')
        self.architecture = architecture
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.net = network(ARCHITECTURES[architecture], 6)

    def forward(self, observations):
        return self.net(observations)

    def probabilities(self, observations):
        return self(observations).softmax(-1)

    def sample(self, observations, generator):
        probabilities = self.probabilities(observations)
        return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


class Value(nn.Module):
    """Separate action-independent MC baseline, never sharing actor parameters."""
    def __init__(self, seed=0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.net = network((64, 64), 1)

    def forward(self, observations):
        return self.net(observations).squeeze(-1)


def load_legacy(path):
    """Copy a frozen original PolicyNet state into a v2 original MLP."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    policy = Policy('original')
    policy.load_state_dict({f'net.{layer}.{kind}': state[f'{old}.{kind}']
                            for layer, old in ((0, 'hidden'), (2, 'out'))
                            for kind in ('weight', 'bias')})
    return policy


class Uniform(nn.Module):
    """Uniform actions, distinct from a randomly initialized neural policy."""
    def forward(self, observations):
        return torch.zeros((*observations.shape[:-1], 6))


class HandDesigned(nn.Module):
    """Observation-only heuristic: urgent care, hungry foraging, then nesting/breeding.

    No access to hidden world state. A fallible descriptive baseline, not an oracle.
    Direction chooses the largest coordinate, with horizontal ties. Exploration is
    uniform when no goal is visible. Strong preferences retain small stochasticity.
    """
    def forward(self, observations):
        scores = torch.zeros((*observations.shape[:-1], 6))
        flat = observations.reshape(-1, 12)
        for row, x in zip(scores.reshape(-1, 6), flat):
            goal = None
            if x[11] > 0 and x[3] > .25:
                goal = (x[9], x[10])
            elif x[2] > 0 and x[3] < .7:
                goal = (x[0], x[1])
            elif x[3] >= .6 and x[8] > 0:
                goal = (x[6], x[7])
            if goal is not None:
                dx, dy = goal
                action = (3 if dx > 0 else 4) if abs(dx) >= abs(dy) else (2 if dy > 0 else 1)
                if dx == 0 and dy == 0:
                    action = 5 if x[11] == 0 and x[3] >= .6 else 0
                row[action] = 6
        return scores
