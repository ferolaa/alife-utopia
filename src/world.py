"""The environment the creatures live in.

The world is a grid. It holds food and it holds agents. It knows where everything is, and
it answers local queries like "is there food on this square" and "how many agents are near
this square".

The world does not decide what agents do. That logic lives in agent.py. Keeping the
environment separate from the behaviour means the same world is reused unchanged by every
condition in the ablation study.
"""

from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=32)
def _offsets_nearest_first(radius: int) -> tuple[tuple[int, int], ...]:
    """Every offset within a radius, ordered by how many steps away it is.

    Scanning in this order means the first food found is the nearest food, so the search
    can stop the moment it finds anything instead of examining the whole patch. With food
    reasonably common that turns a scan of eighty one squares into a handful.
    """
    offsets = [
        (ox, oy)
        for ox in range(-radius, radius + 1)
        for oy in range(-radius, radius + 1)
    ]
    offsets.sort(key=lambda o: (abs(o[0]) + abs(o[1])))
    return tuple(offsets)


class World:
    """A grid of squares holding food and agents."""

    def __init__(
        self,
        width: int,
        height: int,
        n_food: int,
        food_unlimited: bool = False,
        food_respawn_rate: float = 0.0,
        wrap_edges: bool = True,
        rng: random.Random | None = None,
    ):
        self.width = width
        self.height = height
        self.food_unlimited = food_unlimited
        self.food_respawn_rate = food_respawn_rate
        self.wrap_edges = wrap_edges
        self.rng = rng or random.Random()

        # Food is stored as a set of squares. Food is always sparse here, so a set gives
        # constant time lookups for "is there food on this square".
        self.food: set[tuple[int, int]] = set()

        # Feeding sites. When present, food only appears near these, instead of anywhere
        # on the grid. Calhoun's animals ate at fixed hoppers, and he considered what
        # followed to be the central mechanism of the whole experiment: they learned to
        # associate eating with company and began piling into one feeding area while other
        # parts of the pen stood empty. He called it the behavioural sink. Food scattered
        # evenly across a grid cannot produce it, because there is nowhere in particular to
        # gather.
        self.feeders: set[tuple[int, int]] = set()

        # Nest sites. A fixed set of squares where young can be raised. Empty unless the
        # condition being run enables them. When they exist they are the scarce resource
        # that agents compete over, which is closer to the original pen than food scarcity
        # is, since food there was never the constraint.
        self.nests: set[tuple[int, int]] = set()
        self.occupied_nests: set[tuple[int, int]] = set()

        # The simulation puts agents in here. The world only needs their positions.
        self.agents: list = []

        # An index from square to the agents standing on it. It is rebuilt once per tick,
        # so counting neighbours is a local lookup instead of a scan over every agent.
        self._occupancy: dict[tuple[int, int], list] = {}

        # Neighbour counts for every square, rebuilt each tick alongside the index.
        self._crowding = None

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

    def offset(self, from_x: int, from_y: int, to_x: int, to_y: int) -> tuple[int, int]:
        """The shortest step difference from one square to another.

        On a wrapped grid the direct difference is not always the shortest way. Going three
        steps left can be shorter than going thirty seven steps right. This returns
        whichever is shorter on each axis, which is what an agent should be sensing.
        """
        dx = to_x - from_x
        dy = to_y - from_y
        if self.wrap_edges:
            if dx > self.width // 2:
                dx -= self.width
            elif dx < -(self.width // 2):
                dx += self.width
            if dy > self.height // 2:
                dy -= self.height
            elif dy < -(self.height // 2):
                dy += self.height
        return dx, dy

    # ---------------------------------------------------------------------- food

    def place_feeders(self, n: int, rng=None) -> None:
        """Put n feeding sites down. Food will only appear around these."""
        rng = rng or self.rng
        while len(self.feeders) < n:
            self.feeders.add(self.random_square())

    def scatter_food(self, n: int, spread: int = 3) -> None:
        """Place n new food items.

        With feeders present, food appears within `spread` squares of one of them, so
        eating means going to where the food is, and where the food is, everyone else is
        too. Without feeders it falls anywhere, which is the older uniform behaviour and is
        kept so the two can be compared.
        """
        attempts = 0
        added = 0
        limit = n * 20
        feeders = list(self.feeders)
        while added < n and attempts < limit:
            attempts += 1
            if feeders:
                fx, fy = feeders[self.rng.randrange(len(feeders))]
                square = self.normalise(
                    fx + self.rng.randint(-spread, spread),
                    fy + self.rng.randint(-spread, spread),
                )
            else:
                square = self.random_square()
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
        """Regrow food. Only used when food is limited.

        food_respawn_rate is the average number of new items per tick, and it can be more
        than one. The whole part is always added. The fractional part is added with a
        matching probability, so that over many ticks the average comes out right.

        This rate sets the carrying capacity of the world. One food item returns
        energy_from_food energy, and each agent burns energy_cost_per_tick every tick, so
        the population the world can support is roughly the rate times the ratio of those
        two numbers.
        """
        if self.food_unlimited:
            return
        whole = int(self.food_respawn_rate)
        if whole:
            self.scatter_food(whole)
        remainder = self.food_respawn_rate - whole
        if remainder and self.rng.random() < remainder:
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
        food = self.food
        width, height, wrap = self.width, self.height, self.wrap_edges

        for ox, oy in _offsets_nearest_first(vision):
            # The wrapping is written out here rather than calling normalise. This loop runs
            # millions of times per run, and at that volume the function call itself was a
            # measurable share of the total runtime.
            if wrap:
                square = ((x + ox) % width, (y + oy) % height)
            else:
                sx = x + ox
                sy = y + oy
                if not (0 <= sx < width and 0 <= sy < height):
                    continue
                square = (sx, sy)
            if square in food:
                dist = abs(ox) + abs(oy)                  # manhattan distance, in steps
                closeness = 1.0 - (dist / (2 * vision))   # one means adjacent, zero far
                return (ox / vision, oy / vision, closeness)
        return (0.0, 0.0, 0.0)

    def scatter_nests(self, n: int, on_perimeter: bool = False, band: int = 3) -> None:
        """Place n nest sites. Called once when the world is built.

        In the real pen the nesting boxes were up in the walls and the food and water were
        down in the middle, so a mouse leaving its litter to eat had to travel. Putting the
        nests around the edge and the feeders inside reproduces that separation, which is
        what turns caring for young and feeding yourself into genuinely rival activities
        rather than two things that can be done in the same spot.
        """
        attempts = 0
        while len(self.nests) < n and attempts < n * 40:
            attempts += 1
            if on_perimeter:
                x, y = self.random_square()
                near_edge = (
                    x < band or x >= self.width - band
                    or y < band or y >= self.height - band
                )
                if not near_edge:
                    continue
            else:
                x, y = self.random_square()
            self.nests.add((x, y))

    def is_free_nest(self, x: int, y: int) -> bool:
        square = (x, y)
        return square in self.nests and square not in self.occupied_nests

    def claim_nest(self, x: int, y: int) -> bool:
        """Take a nest square. Returns False if it is not a nest, or already taken."""
        if not self.is_free_nest(x, y):
            return False
        self.occupied_nests.add((x, y))
        return True

    def release_nest(self, x: int, y: int) -> None:
        self.occupied_nests.discard((x, y))

    def nearest_free_nest_direction(
        self, x: int, y: int, vision: int
    ) -> tuple[float, float, float]:
        """Where the closest unoccupied nest is, in the same form as the food sense.

        Returns zeros when this condition has no nests at all, so an agent in a world
        without nests simply receives no nest signal.
        """
        nests, taken = self.nests, self.occupied_nests
        # No nests at all, or none free anywhere. Either way there is nothing to find, and
        # searching every square in range to discover that is pure waste. Once nests get
        # scarce this is the common case, so the check pays for itself.
        if not nests or len(taken) >= len(nests):
            return (0.0, 0.0, 0.0)
        width, height, wrap = self.width, self.height, self.wrap_edges

        for ox, oy in _offsets_nearest_first(vision):
            if wrap:
                square = ((x + ox) % width, (y + oy) % height)
            else:
                sx = x + ox
                sy = y + oy
                if not (0 <= sx < width and 0 <= sy < height):
                    continue
                square = (sx, sy)
            if square in nests and square not in taken:
                dist = abs(ox) + abs(oy)
                closeness = 1.0 - (dist / (2 * vision))
                return (ox / vision, oy / vision, closeness)
        return (0.0, 0.0, 0.0)

    def count_neighbours(self, x: int, y: int, radius: int, exclude=None) -> int:
        """Count the other agents within radius squares of a position.

        This is the crowding measure, and crowding is what the whole project is about, so it
        is defined once here and reused everywhere rather than recomputed ad hoc.
        """
        # The fast path: read the answer straight out of the grid built this tick.
        # exclude is always the asking agent itself, standing on this very square, so
        # removing it is just subtracting one.
        crowding = self._crowding
        if crowding is not None:
            total = int(crowding[x, y])
            if exclude is not None:
                total -= 1
            return max(total, 0)

        total = 0
        occupancy = self._occupancy
        if not occupancy:
            return 0
        width, height, wrap = self.width, self.height, self.wrap_edges

        for ox, oy in _offsets_nearest_first(radius):
            if wrap:
                square = ((x + ox) % width, (y + oy) % height)
            else:
                sx = x + ox
                sy = y + oy
                if not (0 <= sx < width and 0 <= sy < height):
                    continue
                square = (sx, sy)
            here = occupancy.get(square)
            if here:
                total += len(here)
                if exclude is not None and exclude in here:
                    total -= 1
        return total

    # -------------------------------------------------------------------- upkeep

    def rebuild_occupancy(self, crowding_radius: int | None = None) -> None:
        """Refresh the square to agents index. The simulation calls this once per tick.

        If a crowding radius is given, the neighbour count for every square on the grid is
        worked out at the same time. Counting neighbours separately for each agent means
        repeating almost the same small search hundreds of times per tick. Doing the whole
        grid in one go instead turns that into a couple of dozen array operations, and the
        answers are identical.
        """
        self._occupancy = {}
        for agent in self.agents:
            self._occupancy.setdefault((agent.x, agent.y), []).append(agent)
        self._crowding = (
            self._build_crowding_grid(crowding_radius)
            if crowding_radius is not None
            else None
        )

    def _build_crowding_grid(self, radius: int):
        """A grid where each square holds the number of agents within radius of it."""
        counts = np.zeros((self.width, self.height), dtype=np.int32)
        for (x, y), here in self._occupancy.items():
            counts[x, y] = len(here)

        total = np.zeros_like(counts)
        for ox in range(-radius, radius + 1):
            for oy in range(-radius, radius + 1):
                if self.wrap_edges:
                    total += np.roll(np.roll(counts, ox, axis=0), oy, axis=1)
                else:
                    # Without wrapping, shift the overlapping region only.
                    xs_to = slice(max(0, ox), self.width + min(0, ox))
                    xs_from = slice(max(0, -ox), self.width + min(0, -ox))
                    ys_to = slice(max(0, oy), self.height + min(0, oy))
                    ys_from = slice(max(0, -oy), self.height + min(0, -oy))
                    total[xs_to, ys_to] += counts[xs_from, ys_from]
        return total

    @property
    def population(self) -> int:
        return len(self.agents)

    @property
    def density(self) -> float:
        """Agents per square, across the whole grid."""
        return len(self.agents) / (self.width * self.height)

    def clustering(self, neighbour_counts, radius: int) -> float:
        """How much more crowded agents are than an even spread would make them.

        One means evenly spread. Above one means they are gathering, which is the thing
        Calhoun called the behavioural sink.

        The comparison is against (n - 1) * patch / cells, which is what each agent would
        see if every other agent were placed independently at random: each of the other
        n - 1 agents has a patch/cells chance of landing in view. An earlier version used
        density * patch - 1, which is close for a dense population but collapses towards
        zero for a sparse one and sends the ratio to absurd values in exactly the case of a
        small population in a large pen.
        """
        n = len(self.agents)
        if n < 2 or not neighbour_counts:
            return 0.0
        patch = (2 * radius + 1) ** 2
        expected = (n - 1) * patch / (self.width * self.height)
        if expected <= 0:
            return 0.0
        return (sum(neighbour_counts) / len(neighbour_counts)) / expected
