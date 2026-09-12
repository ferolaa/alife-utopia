"""The environment: a grid world holding food and agents.

The world itself is deliberately "dumb": it knows where food and agents are, and it can
answer local questions like "is there food near this square?" or "how many neighbours are
around this square?". It does not decide what agents do - that lives in agent.py, and the
loop that drives everything lives in the simulation module. Keeping the environment
separate from behaviour means the same world can be reused unchanged across all three
experimental conditions.
"""

from __future__ import annotations

import random


class World:
    """A width x height grid containing food and agents."""

    def __init__(
        self,
        width: int,
        height: int,
        n_food: int,
        food_unlimited: bool = False,
        food_respawn_prob: float = 0.02,
        wrap_edges: bool = True,
        rng: random.Random | None = None,
    ):
        self.width = width
        self.height = height
        self.food_unlimited = food_unlimited
        self.food_respawn_prob = food_respawn_prob
        self.wrap_edges = wrap_edges
        self.rng = rng or random.Random()

        # Food is stored as a set of (x, y) squares. A set (rather than a full grid array)
        # keeps "is there food here?" fast while food stays sparse, which it always is here.
        self.food: set[tuple[int, int]] = set()

        # Agents are appended by the simulation. The world only needs their positions.
        self.agents: list = []

        # Rebuilt once per tick: maps a square to the agents standing on it, so that
        # counting neighbours is a cheap local lookup instead of scanning every agent.
        self._occupancy: dict[tuple[int, int], list] = {}

        self.scatter_food(n_food)

    # ------------------------------------------------------------------ positions

    def random_square(self) -> tuple[int, int]:
        return (self.rng.randrange(self.width), self.rng.randrange(self.height))

    def normalise(self, x: int, y: int) -> tuple[int, int]:
        """Bring a position back inside the grid.

        With wrap_edges the grid is a torus: walking off the right edge brings you back on
        the left. Wrapping is the default because it keeps population density uniform -
        with hard walls, agents pile up in corners, which would create fake density
        differences and muddy exactly the effect we are trying to measure.
        """
        if self.wrap_edges:
            return (x % self.width, y % self.height)
        return (max(0, min(self.width - 1, x)), max(0, min(self.height - 1, y)))

    # ---------------------------------------------------------------------- food

    def scatter_food(self, n: int) -> None:
        """Drop n new food items on random empty squares."""
        attempts = 0
        added = 0
        while added < n and attempts < n * 20:
            square = self.random_square()
            attempts += 1
            if square not in self.food:
                self.food.add(square)
                added += 1

    def has_food(self, x: int, y: int) -> bool:
        return (x, y) in self.food

    def take_food(self, x: int, y: int) -> bool:
        """Eat the food on this square. Returns True if there was any.

        When food is unlimited, eaten food immediately reappears somewhere else, so the
        total amount in the world never drops. That is the "utopia" setting: resources
        are never the thing that limits the population.
        """
        square = (x, y)
        if square not in self.food:
            return False
        self.food.discard(square)
        if self.food_unlimited:
            self.scatter_food(1)
        return True

    def respawn_step(self) -> None:
        """Slow regrowth of food, used when food is *not* unlimited."""
        if self.food_unlimited:
            return
        if self.rng.random() < self.food_respawn_prob:
            self.scatter_food(1)

    # ------------------------------------------------------------------- sensing

    def nearest_food_direction(
        self, x: int, y: int, vision: int
    ) -> tuple[float, float, float]:
        """Look around (x, y) and report where the closest food is.

        Returns (dx, dy, closeness), all roughly in the range -1..1, which is the form the
        agent's brain wants as input. If no food is visible, returns zeros. Agents only see
        a small patch around themselves rather than the whole grid - this is both more
        plausible and much cheaper to compute.
        """
        best = None
        best_dist = None
        for ox in range(-vision, vision + 1):
            for oy in range(-vision, vision + 1):
                square = self.normalise(x + ox, y + oy)
                if square in self.food:
                    dist = abs(ox) + abs(oy)          # steps needed on a grid
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best = (ox, oy)
        if best is None:
            return (0.0, 0.0, 0.0)
        dx, dy = best
        closeness = 1.0 - (best_dist / (2 * vision))  # 1 = right here, 0 = at vision edge
        return (dx / vision, dy / vision, closeness)

    def count_neighbours(self, x: int, y: int, radius: int, exclude=None) -> int:
        """How many other agents are within `radius` squares of (x, y).

        This is the crowding measure that both the Calhoun and Freedman conditions hinge
        on, so it is defined once here and reused everywhere.
        """
        total = 0
        for ox in range(-radius, radius + 1):
            for oy in range(-radius, radius + 1):
                square = self.normalise(x + ox, y + oy)
                for other in self._occupancy.get(square, ()):
                    if other is not exclude:
                        total += 1
        return total

    # -------------------------------------------------------------------- upkeep

    def rebuild_occupancy(self) -> None:
        """Refresh the square -> agents index. Called once per tick by the simulation."""
        self._occupancy = {}
        for agent in self.agents:
            self._occupancy.setdefault((agent.x, agent.y), []).append(agent)

    @property
    def population(self) -> int:
        return len(self.agents)

    @property
    def density(self) -> float:
        """Agents per square. The headline number for the crowding question."""
        return len(self.agents) / (self.width * self.height)
