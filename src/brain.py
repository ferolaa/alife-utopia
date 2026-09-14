"""The small neural network that decides what a creature does.

A few numbers describing what the creature senses go in. A choice of action comes out.

The network is tiny on purpose. Its weights are not trained with backpropagation. They are
evolved instead. A creature that lives long enough to reproduce passes a mutated copy of
its weights to its child, and selection does the rest. Small networks evolve much faster
than large ones, because random mutation is a far weaker search method than gradient
descent.

All the weights live in one flat vector. That is why mutation is a single line. We add
gaussian noise to the whole vector at once.
"""

from __future__ import annotations

import random

import numpy as np

# The inputs to the network. Listing them here makes it obvious what the network has to
# work with.
SENSES = (
    "food_dx",         # is the nearest visible food left or right     (minus one to one)
    "food_dy",         # is it above or below                          (minus one to one)
    "food_closeness",  # how near it is, zero means none in sight      (zero to one)
    "energy",          # the creature's own energy, scaled             (zero to one)
    "crowding",        # how many neighbours are nearby, scaled        (zero to one)
)

# The outputs of the network. The order matters, because the output scores are read in
# this order.
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
    """A feed forward network with one hidden layer. It maps senses to an action."""

    N_INPUTS = len(SENSES)
    N_HIDDEN = 8
    N_OUTPUTS = len(ACTIONS)

    # How many weights make up one brain. Two weight matrices plus two bias vectors.
    N_WEIGHTS = (
        N_INPUTS * N_HIDDEN + N_HIDDEN          # first layer
        + N_HIDDEN * N_OUTPUTS + N_OUTPUTS      # second layer
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
        """A new brain with random weights. This is what the first generation starts with."""
        seed = None if rng is None else rng.randrange(2**32)
        np_rng = np.random.default_rng(seed)
        return cls(np_rng.normal(0.0, 1.0, size=cls.N_WEIGHTS))

    def mutated_copy(self, mutation_std: float, rng: random.Random | None = None) -> "Brain":
        """A child's brain. It is this brain plus gaussian noise.

        mutation_std is the standard deviation of that noise. Too small and the population
        never explores anything new. Too large and good solutions are destroyed as fast as
        they appear. This is the most important evolution parameter in the project.
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
        """Run the senses through the network. Returns one score per action."""
        x = np.asarray(senses, dtype=np.float64)
        W1, b1, W2, b2 = self._unpack()
        hidden = np.tanh(x @ W1 + b1)      # tanh keeps activations between minus one and one
        return hidden @ W2 + b2

    def decide(self, senses, rng: random.Random | None = None, temperature: float = 1.0) -> str:
        """Choose an action.

        The scores are turned into probabilities with a softmax, and one action is sampled
        from them. We do not always take the highest scoring action. Here is why. A
        creature that sees no food gets the same senses every tick. With a deterministic
        choice it would repeat the same move forever and only ever sweep one row of the
        grid. Sampling makes it wander, and wandering is how it finds food.

        temperature controls how random the choice is. Low values approach always picking
        the best action. High values approach picking uniformly at random.
        """
        scores = self.action_scores(senses) / max(temperature, 1e-6)
        scores = scores - scores.max()              # subtract the max to avoid overflow in exp
        probs = np.exp(scores)
        probs /= probs.sum()
        r = (rng or random).random()
        cumulative = 0.0
        for action, p in zip(ACTIONS, probs):
            cumulative += p
            if r <= cumulative:
                return action
        return ACTIONS[-1]
