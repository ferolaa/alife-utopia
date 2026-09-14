"""A single creature. A body with energy, carrying a brain.

The agent sits between the world and the brain. Each tick it asks the world what it can
sense, passes those senses to its brain, then carries out whatever the brain chose.

Deciding is the brain's job. Paying the energy cost and living with the outcome is the
agent's job.
"""

from __future__ import annotations

import random

from brain import MOVES, Brain, make_senses


class Agent:
    """One creature."""

    __slots__ = (
        "x", "y", "energy", "brain", "age", "alive", "children",
        "dependents", "dependent_until", "neglect_ticks", "nest",
    )

    def __init__(self, x: int, y: int, energy: float, brain: Brain):
        self.x = x
        self.y = y
        self.energy = energy
        self.brain = brain
        self.age = 0
        self.alive = True
        self.children = 0

        # Young this agent is still responsible for. Stays empty unless parental care is
        # switched on for the condition being run.
        self.dependents: list["Agent"] = []

        # Set on a newborn when parental care is on. Until this age it cannot act, and it
        # dies if left unattended for too long.
        self.dependent_until: int = 0
        self.neglect_ticks: int = 0

        # The nest square this agent is occupying, if any. Released when it grows up.
        self.nest: tuple[int, int] | None = None

    @property
    def is_dependent(self) -> bool:
        return self.age < self.dependent_until

    # ------------------------------------------------------------------- sensing

    def sense(self, world, cfg) -> tuple[float, ...]:
        """Collect every input the brain expects, in the order brain.SENSES lists.

        Senses belonging to a mechanism that is switched off come back as zero. An agent in
        a world with no nests simply receives no nest signal, which needs no special case
        anywhere, because the world has no nests to report.
        """
        food_dx, food_dy, food_closeness = world.nearest_food_direction(
            self.x, self.y, cfg.vision
        )
        nest_dx, nest_dy, nest_closeness = world.nearest_free_nest_direction(
            self.x, self.y, cfg.vision
        )
        neighbours = world.count_neighbours(
            self.x, self.y, cfg.crowding_radius, exclude=self
        )
        pup_dx, pup_dy, pup_need = self._sense_pups(world, cfg)

        return make_senses(
            food_dx=food_dx,
            food_dy=food_dy,
            food_closeness=food_closeness,
            energy=min(1.0, self.energy / cfg.energy_max),
            age=min(1.0, self.age / cfg.max_age) if cfg.max_age else 0.0,
            crowding=min(1.0, neighbours / cfg.crowding_scale),
            nest_dx=nest_dx,
            nest_dy=nest_dy,
            nest_closeness=nest_closeness,
            pup_dx=pup_dx,
            pup_dy=pup_dy,
            pup_need=pup_need,
        )

    def _sense_pups(self, world, cfg) -> tuple[float, float, float]:
        """Where this agent's neediest dependent pup is, and how badly it needs attention.

        Only the most neglected pup is reported rather than all of them. A single clear
        signal is something a small network can act on. A list of pups is not.

        Direction is scaled by vision, the same as the food and nest senses, so every
        directional input arrives on the same scale. pup_need rises from zero towards one
        as a pup approaches the point where neglect kills it, which is what gives evolution
        something to respond to.
        """
        if not self.dependents:
            return (0.0, 0.0, 0.0)

        neediest = None
        worst = -1
        for pup in self.dependents:
            if pup.alive and pup.neglect_ticks > worst:
                worst = pup.neglect_ticks
                neediest = pup
        if neediest is None:
            return (0.0, 0.0, 0.0)

        dx, dy = world.offset(self.x, self.y, neediest.x, neediest.y)
        scale = max(cfg.vision, 1)
        need = min(1.0, worst / max(cfg.neglect_tolerance, 1))
        return (
            max(-1.0, min(1.0, dx / scale)),
            max(-1.0, min(1.0, dy / scale)),
            max(need, 0.05),   # a floor, so "I have a pup" is distinguishable from "none"
        )

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
