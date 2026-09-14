"""The small neural network that decides what a creature does.

A vector describing what the creature senses goes in. A choice of action comes out.

The weights are not trained with backpropagation. They are evolved. A creature that lives
long enough to reproduce passes a mutated copy of its weights to its child, and selection
does the rest. Small networks evolve much faster than large ones, because random mutation
is a far weaker search method than gradient descent.

All the weights live in one flat vector, which is why mutation is a single line. We add
gaussian noise to the whole vector at once.
"""

from __future__ import annotations

import random

import numpy as np

# The inputs to the network, in order.
#
# The list is long because the creature has three separate concerns to balance: feeding
# itself, securing a nest, and caring for its young. Each concern needs its own inputs,
# and the interesting behaviour is in how evolution trades them off against each other.
SENSES = (
    # finding food
    "food_dx",          # direction to nearest visible food, left or right  (-1 to 1)
    "food_dy",          # direction to nearest visible food, up or down     (-1 to 1)
    "food_closeness",   # how near it is, zero means none in sight          (0 to 1)
    # the creature's own state
    "energy",           # own energy, scaled                                (0 to 1)
    "age",              # own age as a fraction of maximum lifespan         (0 to 1)
    "crowding",         # neighbours nearby, scaled                         (0 to 1)
    # finding somewhere to raise young
    "nest_dx",          # direction to nearest free nest                    (-1 to 1)
    "nest_dy",
    "nest_closeness",   # zero means no free nest in sight                  (0 to 1)
    # looking after young already born
    "pup_dx",           # direction to own nearest dependent pup            (-1 to 1)
    "pup_dy",
    "pup_need",         # zero means no dependent pup, rises as it is left alone (0 to 1)
)

SENSE_INDEX = {name: i for i, name in enumerate(SENSES)}

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


def make_senses(**values) -> tuple[float, ...]:
    """Build a sense vector by name, with anything unspecified left at zero.

    Tests and behaviour probes use this instead of writing tuples out by hand. Positional
    tuples silently break whenever the sense list changes, and this list has changed once
    already.
    """
    unknown = set(values) - set(SENSES)
    if unknown:
        raise ValueError(f"unknown senses: {sorted(unknown)}")
    return tuple(float(values.get(name, 0.0)) for name in SENSES)


class Brain:
    """A feed forward network with one hidden layer. It maps senses to an action."""

    N_INPUTS = len(SENSES)
    N_HIDDEN = 12
    N_OUTPUTS = len(ACTIONS)

    # How many weights make up one brain. Two weight matrices plus two bias vectors.
    N_WEIGHTS = (
        N_INPUTS * N_HIDDEN + N_HIDDEN          # first layer
        + N_HIDDEN * N_OUTPUTS + N_OUTPUTS      # second layer
    )

    def __init__(self, weights: np.ndarray):
        if weights.shape != (self.N_WEIGHTS,):
            raise ValueError(f"expected {self.N_WEIGHTS} weights, got {weights.shape}")
        self.weights = weights.astype(np.float64)

    # ------------------------------------------------------------------ creation

    @classmethod
    def random(cls, rng: random.Random | None = None) -> "Brain":
        """A new brain with random weights. What the first generation starts with."""
        seed = None if rng is None else rng.randrange(2**32)
        np_rng = np.random.default_rng(seed)
        return cls(np_rng.normal(0.0, 1.0, size=cls.N_WEIGHTS))

    def mutated_copy(self, mutation_std: float, rng: random.Random | None = None) -> "Brain":
        """A child's brain. This brain plus gaussian noise.

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
        hidden = np.tanh(x @ W1 + b1)     # tanh keeps activations between minus one and one
        return hidden @ W2 + b2

    def decide(self, senses, rng: random.Random | None = None, temperature: float = 1.0) -> str:
        """Choose an action.

        The scores are turned into probabilities with a softmax, and one action is sampled
        from them. We do not always take the highest scoring action. Here is why. A
        creature that sees no food gets the same senses every tick. With a deterministic
        choice it would repeat the same move forever and only ever sweep one row of the
        grid. Sampling makes it wander, and wandering is how it finds food.

        temperature controls how random the choice is. Low values approach always taking
        the best action. High values approach picking uniformly at random.
        """
        scores = self.action_scores(senses) / max(temperature, 1e-6)
        scores = scores - scores.max()             # subtract the max to avoid overflow in exp
        probs = np.exp(scores)
        probs /= probs.sum()
        r = (rng or random).random()
        cumulative = 0.0
        for action, p in zip(ACTIONS, probs):
            cumulative += p
            if r <= cumulative:
                return action
        return ACTIONS[-1]
