"""The tiny neural network that decides what a creature does.

This is a plain feed-forward network: a handful of numbers describing what the creature
senses go in, and a choice of action comes out. It is small on purpose - a few dozen
weights - because the weights are not trained with backpropagation. They are *evolved*:
a creature that survives long enough to reproduce passes a slightly mutated copy of its
weights to its offspring, and selection does the rest. Small networks evolve much faster
than large ones, since random mutation is a far less efficient search than gradient
descent.

All the weights live in one flat vector. That is the whole reason mutation is a one-liner
(add gaussian noise to the vector) rather than something that has to walk a layer
structure.
"""

from __future__ import annotations

import random

import numpy as np

# What the creature senses. Keeping this list explicit here (rather than implied by an
# integer) makes it obvious what the network is being asked to work with.
SENSES = (
    "food_dx",         # direction to nearest visible food, left/right   (-1..1)
    "food_dy",         # direction to nearest visible food, up/down      (-1..1)
    "food_closeness",  # how near that food is, 0 = none visible          (0..1)
    "energy",          # own energy, scaled                               (0..1)
    "crowding",        # how many neighbours are nearby, scaled           (0..1)
)

# What the creature can do. Order matters: the network's outputs are read in this order.
ACTIONS = ("stay", "north", "south", "east", "west", "reproduce")

MOVES = {
    "stay": (0, 0),
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
    "reproduce": (0, 0),
}


class Brain:
    """A one-hidden-layer network mapping senses to an action."""

    N_INPUTS = len(SENSES)
    N_HIDDEN = 8
    N_OUTPUTS = len(ACTIONS)

    # Total number of weights: two weight matrices plus two bias vectors.
    N_WEIGHTS = (
        N_INPUTS * N_HIDDEN + N_HIDDEN          # layer 1 weights + biases
        + N_HIDDEN * N_OUTPUTS + N_OUTPUTS      # layer 2 weights + biases
    )

    def __init__(self, weights: np.ndarray):
        if weights.shape != (self.N_WEIGHTS,):
            raise ValueError(
                f"expected {self.N_WEIGHTS} weights, got {weights.shape}"
            )
        self.weights = weights.astype(np.float64)

    # ------------------------------------------------------------------ creation

    @classmethod
    def random(cls, rng: random.Random | None = None) -> "Brain":
        """A brand-new random brain - what the very first generation starts with."""
        seed = None if rng is None else rng.randrange(2**32)
        np_rng = np.random.default_rng(seed)
        return cls(np_rng.normal(0.0, 1.0, size=cls.N_WEIGHTS))

    def mutated_copy(self, mutation_std: float, rng: random.Random | None = None) -> "Brain":
        """A child's brain: this brain plus small random changes.

        mutation_std controls how big those changes are. Too small and the population
        never discovers anything new; too large and good solutions get destroyed as fast
        as they appear. This is the single most important evolution knob in the project.
        """
        seed = None if rng is None else rng.randrange(2**32)
        np_rng = np.random.default_rng(seed)
        noise = np_rng.normal(0.0, mutation_std, size=self.N_WEIGHTS)
        return Brain(self.weights + noise)

    # ------------------------------------------------------------------- thinking

    def _unpack(self):
        """Slice the flat weight vector back into matrices for the forward pass."""
        i, h, o = self.N_INPUTS, self.N_HIDDEN, self.N_OUTPUTS
        w = self.weights
        at = 0
        W1 = w[at:at + i * h].reshape(i, h);  at += i * h
        b1 = w[at:at + h];                    at += h
        W2 = w[at:at + h * o].reshape(h, o);  at += h * o
        b2 = w[at:at + o]
        return W1, b1, W2, b2

    def action_scores(self, senses) -> np.ndarray:
        """Run the senses through the network and return one score per action."""
        x = np.asarray(senses, dtype=np.float64)
        W1, b1, W2, b2 = self._unpack()
        hidden = np.tanh(x @ W1 + b1)      # tanh keeps values in -1..1, a common choice
        return hidden @ W2 + b2

    def decide(self, senses, rng: random.Random | None = None, temperature: float = 1.0) -> str:
        """Pick an action.

        The scores are turned into probabilities (softmax) and one action is *sampled*,
        rather than always taking the highest-scoring one. This matters: with a strictly
        deterministic choice, a creature that sees no food would repeat the same move
        forever and only ever sweep one row of the grid. Sampling makes it wander, which
        is what lets it stumble onto food in the first place.
        """
        scores = self.action_scores(senses) / max(temperature, 1e-6)
        scores = scores - scores.max()              # keeps exp() from overflowing
        probs = np.exp(scores)
        probs /= probs.sum()
        r = (rng or random).random()
        cumulative = 0.0
        for action, p in zip(ACTIONS, probs):
            cumulative += p
            if r <= cumulative:
                return action
        return ACTIONS[-1]
