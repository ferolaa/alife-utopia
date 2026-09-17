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

import math
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

    __slots__ = ("weights", "_W1", "_b1", "_W2", "_b2")

    def __init__(self, weights: np.ndarray):
        if weights.shape != (self.N_WEIGHTS,):
            raise ValueError(f"expected {self.N_WEIGHTS} weights, got {weights.shape}")
        self.weights = weights.astype(np.float64)

        # The weights never change once a brain exists. A child gets a new Brain rather
        # than having its parent's edited. So the matrices are carved out once here instead
        # of being re-sliced on every single decision, which happens millions of times.
        i, h, o = self.N_INPUTS, self.N_HIDDEN, self.N_OUTPUTS
        at = 0
        self._W1 = self.weights[at:at + i * h].reshape(i, h);  at += i * h
        self._b1 = self.weights[at:at + h];                    at += h
        self._W2 = self.weights[at:at + h * o].reshape(h, o);  at += h * o
        self._b2 = self.weights[at:at + o]

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
        """The weight matrices, as carved out when this brain was built."""
        return self._W1, self._b1, self._W2, self._b2

    def hidden_activations(self, senses) -> np.ndarray:
        """What the hidden layer outputs for these senses.

        Normally nobody looks at this: the hidden layer is a means to an end and only the
        action scores matter. The representation analysis looks at it directly, because the
        question there is what each of the twelve units has come to stand for.

        Accepts one sense vector or a whole stack of them, and returns one activation per
        unit in the first case and a row per sense vector in the second.
        """
        x = np.asarray(senses, dtype=np.float64)
        return np.tanh(x @ self._W1 + self._b1)   # tanh keeps activations in minus one to one

    def action_scores(self, senses) -> np.ndarray:
        """Run the senses through the network. Returns one score per action."""
        return self.hidden_activations(senses) @ self._W2 + self._b2

    def without_unit(self, unit: int) -> "Brain":
        """A copy of this brain with one hidden unit silenced.

        Silencing means the unit's output is forced to zero for every input, which is done
        by clearing the weights that carry it to the output layer. Nothing else about the
        brain changes, so whatever behaviour is lost was being carried by that unit.
        """
        if not 0 <= unit < self.N_HIDDEN:
            raise ValueError(f"no hidden unit {unit}")
        weights = self.weights.copy()
        # The second weight matrix is stored row by row, one row per hidden unit, so the
        # unit's outgoing weights are a single contiguous block.
        start = self.N_INPUTS * self.N_HIDDEN + self.N_HIDDEN + unit * self.N_OUTPUTS
        weights[start:start + self.N_OUTPUTS] = 0.0
        return Brain(weights)

    def without_sense(self, sense: int) -> "Brain":
        """A copy of this brain that cannot see one of its senses.

        The weights carrying that input into the hidden layer are cleared, which is the same
        as the input always reading zero. Everything the brain does with its other senses is
        untouched, so whatever behaviour is lost was resting on the one that was removed.
        """
        if not 0 <= sense < self.N_INPUTS:
            raise ValueError(f"no sense {sense}")
        weights = self.weights.copy()
        # The first weight matrix is stored row by row, one row per input.
        start = sense * self.N_HIDDEN
        weights[start:start + self.N_HIDDEN] = 0.0
        return Brain(weights)

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
        # The softmax is done in plain Python rather than with numpy. There are only six
        # actions, and at that size numpy spends more time on its own call overhead than on
        # the arithmetic. Measured, this is roughly three times faster than the numpy
        # version, and it is the single most frequently executed piece of the simulation.
        raw = self.action_scores(senses).tolist()
        t = temperature if temperature > 1e-6 else 1e-6

        biggest = max(raw)
        total = 0.0
        weights = []
        for value in raw:
            w = math.exp((value - biggest) / t)    # shift by the max so exp cannot overflow
            weights.append(w)
            total += w

        target = (rng or random).random() * total
        cumulative = 0.0
        for action, w in zip(ACTIONS, weights):
            cumulative += w
            if target <= cumulative:
                return action
        return ACTIONS[-1]
