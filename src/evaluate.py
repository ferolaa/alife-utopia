"""Checks that ask whether evolution actually produced useful behaviour.

This matters more than it first appears. A stable population is not proof of anything. A
world with plenty of food will keep random agents alive too, so population alone cannot
tell you whether the brains improved. These functions probe the behaviour directly.
"""

from __future__ import annotations

import random

from brain import Brain, make_senses

# Each case gives the brain a clear situation: food is visible in one direction, the agent
# is healthy, and nobody else is nearby. A food seeking brain should move that way.
FOOD_DIRECTION_CASES = {
    "east": make_senses(food_dx=1.0, food_closeness=0.9, energy=0.5),
    "west": make_senses(food_dx=-1.0, food_closeness=0.9, energy=0.5),
    "north": make_senses(food_dy=-1.0, food_closeness=0.9, energy=0.5),
    "south": make_senses(food_dy=1.0, food_closeness=0.9, energy=0.5),
}

MOVE_ACTIONS = ("north", "south", "east", "west")
_MOVES = MOVE_ACTIONS

# With four possible moves, a brain with no preference gets one in four right.
CHANCE_LEVEL = 0.25


def food_seeking_score(brain: Brain, trials: int = 400, seed: int = 0) -> float:
    """How often this brain moves towards food it can see.

    Returns a fraction between zero and one. Reproduce and stay actions are ignored, since
    the question is only about which way the agent goes when it does move. A score near
    CHANCE_LEVEL means the brain ignores the food inputs entirely.
    """
    rng = random.Random(seed)
    correct = total = 0
    for wanted, senses in FOOD_DIRECTION_CASES.items():
        for _ in range(trials):
            action = brain.decide(senses, rng)
            if action in _MOVES:
                total += 1
                if action == wanted:
                    correct += 1
    return correct / max(total, 1)


PUP_DIRECTION_CASES = {
    "east": make_senses(pup_dx=1.0, pup_need=0.8, energy=0.6),
    "west": make_senses(pup_dx=-1.0, pup_need=0.8, energy=0.6),
    "north": make_senses(pup_dy=-1.0, pup_need=0.8, energy=0.6),
    "south": make_senses(pup_dy=1.0, pup_need=0.8, energy=0.6),
}


def pup_seeking_score(brain: Brain, trials: int = 400, seed: int = 0) -> float:
    """How often this brain moves towards a pup that needs attention.

    The parental care equivalent of food_seeking_score. A brain scoring near CHANCE_LEVEL
    ignores its young entirely, and in a world where pups die unattended that is a brain
    whose line ends.

    This matters more than the food probe once food is unlimited, because under those
    conditions food seeking is not under selection at all and its weights drift. Tending
    young is the skill the population is actually being judged on.
    """
    rng = random.Random(seed)
    correct = total = 0
    for wanted, senses in PUP_DIRECTION_CASES.items():
        for _ in range(trials):
            action = brain.decide(senses, rng)
            if action in _MOVES:
                total += 1
                if action == wanted:
                    correct += 1
    return correct / max(total, 1)


def care_under_conflict(brain: Brain, trials: int = 400, seed: int = 0) -> dict:
    """What a brain does when its pup and its next meal are in opposite directions.

    This is the probe that speaks most directly to Calhoun. The collapse he described ran
    through mothers who stopped returning to their litters under pressure. A brain that
    tends its young when nothing else competes for its attention, but abandons them the
    moment something else does, is exactly the failure mode worth measuring.

    Returns how often the pup is chosen, how often the food is, and the difference.
    """
    rng = random.Random(seed)
    to_pup = to_food = total = 0
    # The pup is east, the food is west. Both signals are strong, and the creature is
    # hungry enough for the food to be worth having.
    senses = make_senses(
        pup_dx=1.0, pup_need=0.9,
        food_dx=-1.0, food_closeness=0.9,
        energy=0.35,
    )
    for _ in range(trials):
        action = brain.decide(senses, rng)
        if action in _MOVES:
            total += 1
            if action == "east":
                to_pup += 1
            elif action == "west":
                to_food += 1
    total = max(total, 1)
    return {
        "towards_pup": to_pup / total,
        "towards_food": to_food / total,
        "preference": (to_pup - to_food) / total,   # positive means the pup wins
    }


def crowding_response(brain: Brain, trials: int = 400, seed: int = 0) -> dict:
    """How a brain's choices change when it is crowded rather than alone.

    This is the probe that speaks to the research question. If agents evolve to reproduce
    less when surrounded, that shows up here as a lower reproduce rate in the crowded case.
    The agent is given plenty of energy in both cases, so any difference comes from the
    crowding input and not from starvation.
    """
    rng = random.Random(seed)
    results = {}
    for label, crowding in (("alone", 0.0), ("crowded", 1.0)):
        counts = {"reproduce": 0, "move": 0, "stay": 0}
        senses = make_senses(energy=0.9, crowding=crowding)
        for _ in range(trials):
            action = brain.decide(senses, rng)
            if action == "reproduce":
                counts["reproduce"] += 1
            elif action == "stay":
                counts["stay"] += 1
            else:
                counts["move"] += 1
        results[label] = {k: v / trials for k, v in counts.items()}
    results["reproduce_change"] = (
        results["crowded"]["reproduce"] - results["alone"]["reproduce"]
    )
    return results


def summarise_population(brains, seed: int = 0) -> dict:
    """Average the probes over a whole population of brains."""
    if not brains:
        return {"n": 0, "food_seeking": 0.0, "reproduce_change": 0.0}
    scores = [food_seeking_score(b, seed=seed + i) for i, b in enumerate(brains)]
    pups = [pup_seeking_score(b, seed=seed + i) for i, b in enumerate(brains)]
    conflict = [care_under_conflict(b, seed=seed + i)["preference"] for i, b in enumerate(brains)]
    changes = [crowding_response(b, seed=seed + i)["reproduce_change"] for i, b in enumerate(brains)]
    return {
        "n": len(brains),
        "food_seeking": sum(scores) / len(scores),
        "food_seeking_min": min(scores),
        "food_seeking_max": max(scores),
        "pup_seeking": sum(pups) / len(pups),
        "pup_seeking_min": min(pups),
        "pup_seeking_max": max(pups),
        "care_vs_food": sum(conflict) / len(conflict),
        "reproduce_change": sum(changes) / len(changes),
    }
