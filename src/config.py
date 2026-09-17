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
    nests_on_perimeter: bool = False   # put the nest boxes around the walls, as in the pen
    perimeter_band: int = 3            # how wide that band of wall is

    # ---------------------------- feeding sites -----------------------------------
    # Zero means food falls anywhere. Any higher number concentrates it at that many
    # fixed sites, which is what the real enclosure had and what makes gathering possible.
    #
    # When the nests are around the walls, the feeders are kept out of that band. Otherwise
    # a feeder can land among the nest boxes by chance, and for that run feeding and minding
    # young stop being rival activities, which is the whole reason for separating them.
    n_feeders: int = 0
    feeder_spread: int = 3             # how far from a feeder its food can land

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

    # ------------- a crowd gets between parent and pup -----------------------------
    # As neighbours accumulate around a pup, the distance at which a parent counts as
    # actually tending it shrinks. In an empty pen, being a few squares away is enough. In
    # a crush you have to be right on top of it.
    #
    # This changes what the world does, not what the creature perceives. An earlier version
    # scaled down the pup inputs before they reached the network instead, and that was a
    # mistake: the policy had never been trained on shrunken inputs, so its behaviour on
    # them was arbitrary, and foraging degraded alongside parenting for no reason anyone
    # could point to. Intervening on observations confounds the manipulation with
    # distribution shift. Intervening on dynamics does not.
    crowding_blocks_care: bool = False
    care_radius_decay: float = 0.10    # how fast the effective care distance shrinks

    # ------------- intruders disturb the nest -------------------------------------
    # A pup is harmed by other creatures crowding its nest, not only by its parent being
    # away. This matters because it is the one source of harm a diligent parent cannot
    # prevent: in the real pen, mothers under constant intrusion abandoned young they were
    # otherwise perfectly able to raise.
    nest_intrusion_enabled: bool = False
    intrusion_threshold: int = 6       # neighbours around the nest before it counts

    # -------------------------------------------------------------- the evolution
    mutation_std: float = 0.12         # standard deviation of the mutation noise
    action_temperature: float = 1.0    # softmax temperature when sampling an action

    def variant(self, name: str, **changes) -> "SimConfig":
        """A copy of this config with some parameters changed."""
        return replace(self, name=name, **changes)


DEFAULT = SimConfig()


# The enclosure, set up to match Calhoun's as closely as this model can.
#
# Four breeding pairs went into a pen built for several thousand, with food and water
# freely available and no way out. Nesting boxes were up in the walls; food was down in the
# middle. The population grew for months, levelled off far below the pen's capacity, and
# then declined.
#
# What is reproduced here: unlimited food, no predators or disease, solid walls, nest boxes
# around the perimeter, feeding at a few fixed sites, ageing and maturity, young that die
# unattended, and a small competent founding group in a space that could hold far more.
#
# What is not: dominance hierarchies, individual recognition, and mating behaviour, all of
# which need creatures that can remember specific other creatures.
UNIVERSE_25 = SimConfig(
    name="universe25",

    # A pen, not a doughnut. Walls matter here: animals bunch up in corners and along
    # edges, and that bunching is part of what is being studied rather than an artefact to
    # design away.
    width=48,
    height=48,
    wrap_edges=False,

    # Food is never the constraint, but it comes from a handful of places.
    food_unlimited=True,
    n_food=260,
    n_feeders=4,
    feeder_spread=3,

    # Nest boxes, well short of what the population will need. Scattered, not in the walls,
    # and that is a deliberate choice rather than the obvious one.
    #
    # In the real pen the nest boxes were up in the walls and the food was down in the
    # middle, so leaving a litter to eat meant a journey. Reproducing that here kills the
    # colony every time. The median nest sits 27 steps from the nearest feeder, a pup dies
    # after 25 ticks alone, and a creature can only see four squares, so it cannot find a
    # free nest on the far side of the pen in the first place. Populations founded in that
    # layout go extinct before tick 1000, whatever the timings are set to.
    #
    # The reason is the sensing, not the distance. Calhoun's mice had lived in that pen for
    # months and knew where everything was. These creatures know only what is within four
    # squares of them, and a layout that has to be learned by heart is one they cannot use.
    # Scattering the nests puts a free one within sight often enough for the population to
    # function, which is the condition every other question here depends on.
    nests_enabled=True,
    n_nests=70,
    nests_on_perimeter=False,

    # A small founding group in a large space, as in the original.
    n_initial_agents=16,
    max_population=3000,
    n_ticks=12000,

    ageing_enabled=True,
    parental_care_enabled=True,
    crowding_cost_enabled=True,
)
