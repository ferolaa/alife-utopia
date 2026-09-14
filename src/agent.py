"""A single creature. A body with energy, carrying a brain.

The agent sits between the world and the brain. Each tick it asks the world what it can
sense, passes those senses to its brain, then carries out whatever the brain chose.

Deciding is the brain's job. Paying the energy cost and living with the outcome is the
agent's job.
"""

from __future__ import annotations

import random

from brain import MOVES, Brain


class Agent:
    """One creature."""

    __slots__ = ("x", "y", "energy", "brain", "age", "alive", "children")

    def __init__(self, x: int, y: int, energy: float, brain: Brain):
        self.x = x
        self.y = y
        self.energy = energy
        self.brain = brain
        self.age = 0
        self.alive = True
        self.children = 0

    # ------------------------------------------------------------------- sensing

    def sense(self, world, cfg) -> tuple[float, float, float, float, float]:
        """Collect the five inputs the brain expects, in the order brain.SENSES lists."""
        food_dx, food_dy, food_closeness = world.nearest_food_direction(
            self.x, self.y, cfg.vision
        )
        energy_level = min(1.0, self.energy / cfg.energy_max)
        neighbours = world.count_neighbours(
            self.x, self.y, cfg.crowding_radius, exclude=self
        )
        crowding = min(1.0, neighbours / cfg.crowding_scale)
        return (food_dx, food_dy, food_closeness, energy_level, crowding)

    # -------------------------------------------------------------------- acting

    def act(self, world, cfg, rng: random.Random) -> "Agent | None":
        """Live one tick. Returns a child if the agent reproduced, otherwise None.

        The order of the steps is deliberate, and worth stating in the report. Costs are
        paid first, then the action is taken, then death is checked. So an agent can
        starve on the same tick it found food. That keeps the energy budget honest and
        stops agents from escaping death by feeding at the last moment.
        """
        self.age += 1

        # 1. The metabolic cost of staying alive.
        self.energy -= cfg.energy_cost_per_tick

        # 2. The cost of nearby agents, if this condition enables it.
        if cfg.crowding_cost_enabled:
            neighbours = world.count_neighbours(
                self.x, self.y, cfg.crowding_radius, exclude=self
            )
            self.energy -= neighbours * cfg.crowding_energy_cost

        # 3. Sense, decide, act.
        senses = self.sense(world, cfg)
        action = self.brain.decide(senses, rng, cfg.action_temperature)

        child = None
        if action == "reproduce":
            child = self._try_reproduce(world, cfg, rng)
        else:
            dx, dy = MOVES[action]
            if dx or dy:
                self.x, self.y = world.normalise(self.x + dx, self.y + dy)

        # 4. Eating is automatic. Standing on food means eating it. So the brain only has
        #    to learn to reach food. It does not also have to learn a separate eat action.
        if world.take_food(self.x, self.y):
            self.energy = min(cfg.energy_max, self.energy + cfg.energy_from_food)

        # 5. No energy left means death.
        if self.energy <= 0:
            self.alive = False

        return child

    def _try_reproduce(self, world, cfg, rng: random.Random) -> "Agent | None":
        """Spend energy to place a child with a mutated brain on a nearby square.

        Below the energy threshold this fails and the tick is wasted. That is itself a
        selection pressure. Brains that attempt to reproduce while starving do worse.
        """
        if self.energy < cfg.reproduce_threshold:
            return None

        self.energy -= cfg.reproduce_cost
        self.children += 1

        dx, dy = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)])
        cx, cy = world.normalise(self.x + dx, self.y + dy)
        return Agent(
            x=cx,
            y=cy,
            energy=cfg.reproduce_cost,     # the energy the parent spent goes to the child
            brain=self.brain.mutated_copy(cfg.mutation_std, rng),
        )
