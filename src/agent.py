"""A single creature. A body with energy, carrying a brain.

The agent sits between the world and the brain. Each tick it asks the world what it can
sense, passes those senses to its brain, then carries out whatever the brain chose.

Deciding is the brain's job. Paying the energy cost and living with the outcome is the
agent's job.
"""

from __future__ import annotations

import random

from brain import MOVES, Brain, make_senses


def _within(world, one, other, radius: float) -> bool:
    """Is one creature close enough to the other to count as present?

    Distance is measured as the larger of the two axis gaps, which makes the area a square
    of side twice the radius. That matches how every other range in the simulation is
    measured, so care range and crowding range mean the same kind of thing.
    """
    dx, dy = world.offset(one.x, one.y, other.x, other.y)
    return max(abs(dx), abs(dy)) <= radius


def draw_lifespan(cfg, rng: random.Random) -> float:
    """How long this particular creature gets to live.

    Lifespans vary between individuals rather than every creature dying at exactly the same
    age. Without that spread, a generation born together would also die together, producing
    artificial waves of death that have nothing to do with anything we are studying.

    When ageing is switched off the lifespan is infinite, which is how the ablation runs
    remove this mechanism without any special cases elsewhere.
    """
    if not cfg.ageing_enabled:
        return float("inf")
    spread = cfg.max_age_spread
    return cfg.max_age * rng.uniform(1.0 - spread, 1.0 + spread)


class Agent:
    """One creature."""

    __slots__ = (
        "x", "y", "energy", "brain", "age", "alive", "children",
        "dependents", "dependent_until", "neglect_ticks", "nest", "neighbours",
        "lifespan", "parent", "cause_of_death", "neglect_suffered", "impairment",
        "last_senses",
    )

    def __init__(
        self,
        x: int,
        y: int,
        energy: float,
        brain: Brain,
        lifespan: float = float("inf"),
    ):
        self.x = x
        self.y = y
        self.energy = energy
        self.brain = brain
        self.age = 0
        self.alive = True
        self.children = 0

        # The age this creature dies at. Infinite when ageing is switched off.
        self.lifespan = lifespan

        # Young this agent is still responsible for. Stays empty unless parental care is
        # switched on for the condition being run.
        self.dependents: list["Agent"] = []

        # Set on a newborn when parental care is on. Until this age it cannot act, and it
        # dies if left unattended for too long.
        self.dependent_until: int = 0
        self.neglect_ticks: int = 0
        self.parent: "Agent | None" = None

        # Total ticks this creature spent unattended as a pup. Unlike neglect_ticks, which
        # falls again when a parent returns, this only ever goes up. Being rescued keeps a
        # pup alive; it does not undo the time it already spent alone.
        self.neglect_suffered: int = 0

        # The sense vector this creature last acted on. Kept so that a trainer outside the
        # simulation can see what the creature saw when it made a choice, which is what
        # policy gradient methods need and evolution does not.
        self.last_senses = None

        # How badly its upbringing damaged it, from zero to one. Set once, when it grows up.
        # It scales down how well this creature can sense its own young, so a badly raised
        # creature is a bad parent even with a perfectly good brain.
        self.impairment: float = 0.0

        # Recorded when the creature dies, so the run can be broken down by what actually
        # killed things. Telling starvation apart from neglect is the whole point of the
        # experiment, and a bare death count cannot do that.
        self.cause_of_death: str | None = None

        # The nest square this agent is occupying, if any. Released when it grows up.
        self.nest: tuple[int, int] | None = None

        # How many neighbours this agent had when it last acted. Counting neighbours is the
        # most expensive thing in the simulation, so it is done once per tick and the
        # answer is reused by the crowding cost, the senses and the metrics. None means
        # this agent has not acted yet, which is true of newborns on the tick they appear.
        self.neighbours: int | None = None

    @property
    def is_dependent(self) -> bool:
        return self.age < self.dependent_until

    # ------------------------------------------------------------------- sensing

    def sense(self, world, cfg, neighbours: int | None = None) -> tuple[float, ...]:
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
        if neighbours is None:
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

        # A creature damaged by its own upbringing perceives its young only faintly. At full
        # impairment it cannot sense them at all, and so has no reason to go back to them.
        # Its brain is untouched. What it lost was the childhood that would have let the
        # brain do its job.
        clarity = 1.0 - self.impairment

        return (
            max(-1.0, min(1.0, dx / scale)) * clarity,
            max(-1.0, min(1.0, dy / scale)) * clarity,
            max(need, 0.05) * clarity,
        )

    # -------------------------------------------------------------------- acting

    def act(self, world, cfg, rng: random.Random, action: str | None = None) -> "Agent | None":
        """Live one tick. Returns a child if the agent reproduced, otherwise None.

        If an action is supplied, the creature performs it instead of consulting its own
        brain. That is how a policy trained outside the simulation drives the creature: the
        trainer chooses for every creature at once, in a single batched forward pass, rather
        than each creature being asked separately.

        The order of the steps is deliberate, and worth stating in the report. Costs are
        paid first, then the action is taken, then death is checked. So an agent can
        starve on the same tick it found food. That keeps the energy budget honest and
        stops agents from escaping death by feeding at the last moment.
        """
        self.age += 1

        # A nest is held only while the young one in it still needs it. Once it has grown
        # past that age the nest goes back into circulation. Nests are the scarce resource
        # in this world, so holding one longer than necessary would quietly strangle the
        # whole population.
        if self.nest is not None and self.age >= cfg.dependency_ticks:
            world.release_nest(*self.nest)
            self.nest = None

        # A pup that is still dependent does not forage, move or decide anything. It is
        # being fed. The only thing that matters to it is whether a parent is nearby.
        if self.is_dependent:
            self._spend_tick_as_pup(world, cfg)
            return None

        # The tick it stops being dependent, it stops being its parent's problem, and
        # whatever its upbringing did to it is settled.
        if self.parent is not None:
            self._grow_up(cfg)

        # Count neighbours once, here, and reuse the answer everywhere else this tick.
        self.neighbours = world.count_neighbours(
            self.x, self.y, cfg.crowding_radius, exclude=self
        )

        # 1. The metabolic cost of staying alive.
        self.energy -= cfg.energy_cost_per_tick

        # 2. The cost of nearby agents, if this condition enables it.
        if cfg.crowding_cost_enabled:
            self.energy -= self.neighbours * cfg.crowding_energy_cost

        # 3. Sense, decide, act.
        self.last_senses = self.sense(world, cfg, neighbours=self.neighbours)
        if action is None:
            action = self.brain.decide(self.last_senses, rng, cfg.action_temperature)

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

        # 5. Death, from either starvation or old age.
        if self.energy <= 0:
            self.alive = False
            self.cause_of_death = "starvation"
        elif self.age >= self.lifespan:
            self.alive = False
            self.cause_of_death = "old age"

        return child

    def _spend_tick_as_pup(self, world, cfg) -> None:
        """One tick in the life of a helpless newborn.

        The pup cannot feed itself, so nothing it does affects its survival. What decides
        whether it lives is whether its parent stays close enough, often enough. Time spent
        unattended accumulates, and time spent attended pays it back down again, so a parent
        that comes and goes can still raise a pup, while one that wanders off for good
        cannot.

        This is the mechanism the real collapse ran through. Mothers under pressure stopped
        returning to their litters, the litters did not survive, and a population that
        cannot raise its young has no future however many adults it currently has.
        """
        parent = self.parent

        # How close the parent has to be for its presence to do any good. A crowd around
        # the pup shrinks this, so the same parent standing in the same place stops
        # counting as present once enough bodies are between them.
        reach = cfg.care_radius
        if cfg.crowding_blocks_care:
            crowd = world.count_neighbours(
                self.x, self.y, cfg.crowding_radius, exclude=self
            )
            reach = cfg.care_radius / (1.0 + cfg.care_radius_decay * crowd)

        attended = (
            parent is not None
            and parent.alive
            and not parent.is_dependent
            and _within(world, parent, self, reach)
        )

        # Too many bodies around the nest and the pup suffers whatever its parent does.
        # This is the only harm in the model that diligent parenting cannot prevent, which
        # is precisely why it is worth testing: every other pressure here has an escape.
        if cfg.nest_intrusion_enabled and attended:
            intruders = world.count_neighbours(
                self.x, self.y, cfg.crowding_radius, exclude=self
            )
            if parent is not None and _within(world, parent, self, cfg.crowding_radius):
                intruders -= 1          # the parent is not an intruder
            if intruders >= cfg.intrusion_threshold:
                attended = False

        if attended:
            if self.neglect_ticks > 0:
                self.neglect_ticks -= 1
        else:
            self.neglect_ticks += 1
            self.neglect_suffered += 1
            if self.neglect_ticks > cfg.neglect_tolerance:
                self.alive = False
                self.cause_of_death = "neglect"

    def _grow_up(self, cfg) -> None:
        """Leave the parent, and carry forward whatever the upbringing cost.

        Impairment is the fraction of its infancy this creature spent alone, so a pup that
        was always attended is unharmed and one that was mostly alone is badly harmed. The
        harm is permanent. There is no recovering from it later.
        """
        if cfg.developmental_damage_enabled and cfg.dependency_ticks > 0:
            share_alone = self.neglect_suffered / cfg.dependency_ticks
            self.impairment = min(1.0, cfg.damage_scale * share_alone)
        self._leave_the_parent()

    def _leave_the_parent(self) -> None:
        """The parent is no longer responsible for this creature."""
        parent = self.parent
        if parent is not None:
            try:
                parent.dependents.remove(self)
            except ValueError:
                pass
        self.parent = None

    def refund_birth(self, cfg) -> None:
        """Undo a birth the simulation could not accept.

        The population cap is a safety limit on the program, not a fact about the world, so
        it must not quietly destroy energy or count as a failed breeding attempt. Without
        this the population sits at the cap while every parent pays for children that never
        exist, which drains the whole population.
        """
        self.energy = min(cfg.energy_max, self.energy + cfg.reproduce_cost)
        self.children -= 1
        if self.dependents:
            # The child that was refused was the most recent one added.
            ghost = self.dependents.pop()
            ghost.parent = None

    def _try_reproduce(self, world, cfg, rng: random.Random) -> "Agent | None":
        """Spend energy to place a child with a mutated brain on a nearby square.

        Below the energy threshold this fails and the tick is wasted. That is itself a
        selection pressure. Brains that attempt to reproduce while starving do worse.
        """
        if self.energy < cfg.reproduce_threshold:
            return None

        # Too young or too old to breed. Maturity matters because a newborn that can breed
        # at once makes population growth far too fast to resemble anything real, and the
        # upper limit matters because a population where nobody ages out of breeding can
        # always recover, which would rule out the collapse before we start looking for it.
        if cfg.ageing_enabled and not (cfg.maturity_age <= self.age <= cfg.fertility_end_age):
            return None

        # Young can only be raised in a nest, and there are not enough nests to go round.
        # The parent must be standing on a free one. This is what turns space itself into
        # something worth competing over, which is the situation the original enclosure was
        # really in: food was never short, but good places to raise young were.
        nest_square = None
        if cfg.nests_enabled:
            if not world.claim_nest(self.x, self.y):
                return None
            nest_square = (self.x, self.y)

        self.energy -= cfg.reproduce_cost
        self.children += 1

        if nest_square is not None:
            # The child stays in the nest it was born in.
            cx, cy = nest_square
        else:
            dx, dy = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)])
            cx, cy = world.normalise(self.x + dx, self.y + dy)

        child = Agent(
            x=cx,
            y=cy,
            energy=cfg.reproduce_cost,     # the energy the parent spent goes to the child
            brain=self.brain.mutated_copy(cfg.mutation_std, rng),
            lifespan=draw_lifespan(cfg, rng),
        )
        child.nest = nest_square

        if cfg.parental_care_enabled:
            child.dependent_until = cfg.dependency_ticks
            child.parent = self
            self.dependents.append(child)

        return child
