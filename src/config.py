"""Every parameter of the simulation, kept in one place.

This matters for this project in particular. Every condition in the ablation study differs
only in these numbers. No logic differs between them at all. That is what makes the
comparison between conditions a fair one, and it is why each mechanism is switched on and
off by a flag here rather than by editing the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SimConfig:
    name: str = "default"

    # ----------------------------------------------------------------- the world
    width: int = 40
    height: int = 40
    n_food: int = 150
    food_unlimited: bool = False       # True means eaten food reappears immediately
    food_respawn_rate: float = 4.0     # average new food items per tick, when limited
    wrap_edges: bool = True            # True makes the grid a torus

    # ------------------------------------------------------------- the simulation
    n_ticks: int = 3000
    n_initial_agents: int = 40
    max_population: int = 1500         # a safety cap, so a runaway population cannot hang
    seed: int = 0

    # ---------------------------------------------------------------- the senses
    vision: int = 4                    # how many squares away food can be seen
    crowding_radius: int = 2           # how close another agent must be to count
    crowding_scale: float = 8.0        # neighbour count that reads as fully crowded

    # ------------------------------------------------------------ the energy rules
    energy_start: float = 25.0
    energy_max: float = 60.0           # a cap, so agents cannot hoard energy forever
    energy_cost_per_tick: float = 0.25  # the metabolic cost of staying alive
    energy_from_food: float = 12.0
    reproduce_threshold: float = 35.0  # below this, attempting to reproduce does nothing
    reproduce_cost: float = 18.0       # energy transferred from parent to child

    # ------------------------- the cost of unavoidable social contact -------------
    # When enabled, each nearby agent drains a little energy per tick. This represents
    # social contact being costly in itself, as opposed to density simply being high. It is
    # the point Freedman raised against Calhoun, and it is one of the mechanisms we ablate.
    crowding_cost_enabled: bool = False

    # Per neighbour, per tick. This number is calibrated rather than picked: at the density
    # the population actually reaches, around ten to fifteen neighbours each, it should cost
    # roughly what staying alive costs, so that crowding is a serious pressure when dense
    # and close to nothing when sparse. That asymmetry is the whole point, and it is what
    # Calhoun described. At the original value of 0.12 a crowded creature paid seven times
    # its own metabolism, which no population can survive at any density, so the result was
    # decided by the parameter rather than by anything the creatures did.
    crowding_energy_cost: float = 0.025

    # ---------------------------- ageing, the first Calhoun mechanism -------------
    # Without ageing a population can never decline the way Calhoun's did, because nothing
    # ever dies of old age. Maturity matters too: a newborn that can breed immediately
    # makes the population dynamics far too fast.
    ageing_enabled: bool = False
    max_age: int = 900                 # an agent dies once it passes this age
    max_age_spread: float = 0.2        # lifespans vary by this fraction, so deaths spread out
    maturity_age: int = 120            # cannot reproduce before this age
    fertility_end_age: int = 700       # cannot reproduce after this age

    # ---------------------------- nest sites, the second mechanism ----------------
    # A fixed number of squares where young can be raised. Nests are what agents compete
    # over once food is unlimited.
    nests_enabled: bool = False
    n_nests: int = 60

    # ---------------------------- parental care, the third mechanism --------------
    # Newborns are helpless. They cannot act, and they die if no parent stays close. This
    # is the mechanism whose breakdown drove the real collapse, through neglected litters
    # that never grew into functioning adults.
    parental_care_enabled: bool = False
    dependency_ticks: int = 60         # how long a pup stays helpless
    care_radius: int = 3               # how close a parent counts as present
    neglect_tolerance: int = 25        # ticks a pup survives unattended before dying

    # ------------- developmental damage, the fourth mechanism ---------------------
    # A pup that survives being left alone does not come out of it unharmed. It grows into
    # an adult that is worse at noticing and responding to its own young.
    #
    # This is the one mechanism in the model that can carry damage forward in time. Every
    # other pressure here is self correcting: when crowding hurts, creatures die, and fewer
    # creatures means less crowding. Damage that outlives the conditions that caused it
    # works the other way round, because a badly raised generation raises the next one worse
    # still, whether or not the crowding that started it is still there.
    #
    # Calhoun described exactly this. The neglected pups of Universe 25 did not all die.
    # Many grew up, and grew up unable to raise young of their own.
    #
    # Note that this is not inherited. The brain a pup is born with is untouched. What
    # damages it is its own upbringing, which is why the effect spreads through care rather
    # than through genes.
    developmental_damage_enabled: bool = False
    damage_scale: float = 1.0          # how strongly time spent unattended translates to harm

    # -------------------------------------------------------------- the evolution
    mutation_std: float = 0.12         # standard deviation of the mutation noise
    action_temperature: float = 1.0    # softmax temperature when sampling an action

    def variant(self, name: str, **changes) -> "SimConfig":
        """A copy of this config with some parameters changed."""
        return replace(self, name=name, **changes)


DEFAULT = SimConfig()
