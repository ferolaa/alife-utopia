"""What the twelve hidden units of a trained brain have come to stand for.

Everything else in this project measures behaviour: does a creature go to its pup, does the
population cluster, does it collapse. This module looks inside the network instead.

Two questions, and they check each other.

The first is what each unit responds to. Run a large sample of real situations through the
network, record what each unit outputs, and see which senses that output tracks. A unit
whose activation rises and falls with pup_need is, in some loose sense, about pups.

The second is what each unit is for. Silence it and re-measure the behaviour. A unit that
looks like it is about pups but whose removal costs nothing was not carrying the behaviour.
Correlation says what a unit listens to; silencing says what depends on it. Only where the
two agree is there anything worth claiming.

The samples have to come from a real run rather than from random numbers. The senses a
creature actually receives are nothing like uniform: most of the time no pup needs anything,
energy sits in a narrow band, and the directional senses are dominated by wherever the food
happens to be. A unit tuned to a situation that never arises is not doing any work, and
random inputs would hide that.
"""

from __future__ import annotations

import numpy as np

from brain import ACTIONS, SENSES, Brain
from evaluate import (care_under_conflict, crowding_response, food_seeking_score,
                      pup_seeking_score)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Plain correlation, with the flat cases answered rather than left as nan.

    A sense that never varies, or a unit that is saturated and never moves, has no
    correlation with anything. Numpy reports that as nan and then warns about it. Zero is
    the honest answer and it keeps the tables readable.
    """
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def tuning(brain: Brain, senses_matrix) -> list[dict]:
    """For every hidden unit, what it tracks and how much it moves.

    senses_matrix is one row per situation. Returns one entry per unit with:

      activity   how far the unit swings across the sample. A unit pinned at one value
                 carries no information however large its weights are.
      senses     the correlation between the unit's output and each sense.
      listens_to the sense it tracks most strongly.
      votes_for  how strongly the unit pushes each action, weighted by how much it actually
                 moves. A large outgoing weight on a unit that never varies changes nothing.
    """
    X = np.asarray(senses_matrix, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != Brain.N_INPUTS:
        raise ValueError(f"expected a matrix with {Brain.N_INPUTS} columns, got {X.shape}")

    H = brain.hidden_activations(X)
    out = []
    for unit in range(Brain.N_HIDDEN):
        activations = H[:, unit]
        senses = {name: _correlation(X[:, i], activations) for i, name in enumerate(SENSES)}
        listens_to = max(senses, key=lambda name: abs(senses[name]))
        swing = float(activations.std())
        weights = brain._W2[unit]
        out.append({
            "unit": unit,
            "mean": float(activations.mean()),
            "activity": swing,
            "low": float(activations.min()),
            "high": float(activations.max()),
            "senses": senses,
            "listens_to": listens_to,
            "listens_strength": senses[listens_to],
            "votes_for": {action: float(weights[i] * swing)
                          for i, action in enumerate(ACTIONS)},
        })
    return out


def probe(brain: Brain, trials: int = 800, seed: int = 0) -> dict:
    """The four behaviour measures, gathered in one place so they can be compared as a set.

    These are the same probes used everywhere else in the project, so a number here means
    the same thing it means in the experiments.
    """
    return {
        "food_seeking": food_seeking_score(brain, trials=trials, seed=seed),
        "pup_seeking": pup_seeking_score(brain, trials=trials, seed=seed),
        "care_vs_food": care_under_conflict(brain, trials=trials, seed=seed)["preference"],
        "reproduce_change": crowding_response(brain, trials=trials, seed=seed)["reproduce_change"],
    }


def ablation(brain: Brain, trials: int = 800, seed: int = 0) -> dict:
    """Silence each hidden unit in turn and see which behaviours go with it.

    Every probe runs from the same seed, so the intact brain and the silenced one meet the
    same sequence of random draws. Without that, small differences would be swamped by the
    sampling noise in the probes themselves.

    Returns the intact scores and, per unit, the change from them. A large negative change
    in one behaviour and nothing elsewhere is a unit that carries that behaviour on its own.
    """
    intact = probe(brain, trials=trials, seed=seed)
    units = []
    for unit in range(Brain.N_HIDDEN):
        silenced = probe(brain.without_unit(unit), trials=trials, seed=seed)
        units.append({
            "unit": unit,
            "scores": silenced,
            "change": {k: silenced[k] - intact[k] for k in intact},
        })
    return {"intact": intact, "units": units}


def specialisation(ablation_result: dict) -> float:
    """How concentrated a behaviour is, on a scale from spread out to carried by one unit.

    Take the damage each unit does to pup seeking, and ask what share of the total damage
    the worst single unit accounts for. One means a single unit holds the behaviour and the
    rest contribute nothing. A twelfth means every unit matters equally.

    This is the number that makes two policies comparable. Comparing raw weights across
    networks trained from different seeds is meaningless, because units are in no particular
    order and nothing lines them up.
    """
    damage = [max(0.0, -u["change"]["pup_seeking"]) for u in ablation_result["units"]]
    total = sum(damage)
    if total <= 0:
        return 0.0
    return max(damage) / total
