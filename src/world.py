"""The environment the creatures live in.

The world is a grid. It holds food and it holds agents. It knows where everything is, and
it answers local queries like "is there food on this square" and "how many agents are near
this square".

The world does not decide what agents do. That logic lives in agent.py. Keeping the
environment separate from the behaviour means the same world is reused unchanged across
all three experimental conditions.
"""

from __future__ import annotations

import random


class World:
    """A grid of squares holding food and agents."""

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

        # Food is stored as a set of squares. Food is always sparse here, so a set gives
        # constant time lookups for "is there food on this square".
        self.food: set[tuple[int, int]] = set()

        # The simulation puts agents in here. The world only needs their positions.
        self.agents: list = []

        # An index from square to the agents standing on it. It is rebuilt once per tick,
        # so counting neighbours is a local lookup instead of a scan over every agent.
        self._occupancy: dict[tuple[int, int], list] = {}

        self.scatter_food(n_food)

    # ------------------------------------------------------------------ positions

    def random_square(self) -> tuple[int, int]:
        return (self.rng.randrange(self.width), self.rng.randrange(self.height))

    def normalise(self, x: int, y: int) -> tuple[int, int]:
        """Bring a position back inside the grid.

        With wrap_edges the grid is a torus. Walking off the right edge brings you back on
        the left. Wrapping is the default because it keeps density uniform. With solid
        walls the agents pile up in the corners, which creates artificial density
        gradients, and density is exactly what this project is trying to measure.
        """
        if self.wrap_edges:
            return (x % self.width, y % self.height)
        return (max(0, min(self.width - 1, x)), max(0, min(self.height - 1, y)))

    # ---------------------------------------------------------------------- food

    def scatter_food(self, n: int) -> None:
        """Place n new food items on random empty squares."""
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
        """Eat the food on this square. Returns True if there was food there.

        When food is unlimited, an eaten item immediately reappears somewhere else, so the
        total never drops. That is the utopia setting. Resources are never the constraint
        on the population.
        """
        square = (x, y)
        if square not in self.food:
            return False
        self.food.discard(square)
        if self.food_unlimited:
            self.scatter_food(1)
        return True

    def respawn_step(self) -> None:
        """Slow regrowth of food. Only used when food is limited."""
        if self.food_unlimited:
            return
        if self.rng.random() < self.food_respawn_prob:
            self.scatter_food(1)

    # ------------------------------------------------------------------- sensing

    def nearest_food_direction(
        self, x: int, y: int, vision: int
    ) -> tuple[float, float, float]:
        """Report where the closest visible food is, relative to a square.

        Returns three values. The first two are the direction to that food on each axis.
        The third is how close it is. All three are roughly between minus one and one,
        which is the scale the brain expects for its inputs. If no food is in range, all
        three are zero.

        Agents only see a small patch around themselves rather than the whole grid. This is
        more plausible, and it keeps the cost per agent per tick constant instead of
        growing with the size of the world.
        """
        best = None
        best_dist = None
        for ox in range(-vision, vision + 1):
            for oy in range(-vision, vision + 1):
                square = self.normalise(x + ox, y + oy)
                if square in self.food:
                    dist = abs(ox) + abs(oy)          # manhattan distance, in grid steps
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best = (ox, oy)
        if best is None:
            return (0.0, 0.0, 0.0)
        dx, dy = best
        closeness = 1.0 - (best_dist / (2 * vision))  # one means adjacent, zero means far
        return (dx / vision, dy / vision, closeness)

    def count_neighbours(self, x: int, y: int, radius: int, exclude=None) -> int:
        """Count the other agents within radius squares of a position.

        This is the crowding measure. Both the Calhoun and the Freedman conditions depend
        on it, so it is defined once here and reused everywhere.
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
        """Refresh the square to agents index. The simulation calls this once per tick."""
        self._occupancy = {}
        for agent in self.agents:
            self._occupancy.setdefault((agent.x, agent.y), []).append(agent)

    @property
    def population(self) -> int:
        return len(self.agents)

    @property
    def density(self) -> float:
        """Agents per square. This is the headline number for the crowding question."""
        return len(self.agents) / (self.width * self.height)
