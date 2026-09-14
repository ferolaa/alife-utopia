"""Every tunable number in the simulation, in one place.

Keeping these out of the simulation code matters for this project specifically: the three
experimental conditions differ *only* in these numbers, not in any logic. That is what
makes the comparison between them fair - nothing else changes.
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
    food_unlimited: bool = False       # True = eaten food instantly reappears elsewhere
    food_respawn_prob: float = 0.3     # chance per tick of one new food item (if limited)
    wrap_edges: bool = True

    # ------------------------------------------------------------- the simulation
    n_ticks: int = 3000
    n_initial_agents: int = 40
    max_population: int = 1500         # safety cap so a runaway population can't hang us
    seed: int = 0

    # ---------------------------------------------------------------- the senses
    vision: int = 4                    # how many squares away food can be seen
    crowding_radius: int = 2           # how far away another creature still counts as near
    crowding_scale: float = 8.0        # neighbours needed to register as "fully crowded"

    # ------------------------------------------------------------ the energy rules
    energy_start: float = 25.0
    energy_max: float = 60.0           # cap, so creatures cannot hoard forever
    energy_cost_per_tick: float = 0.25  # the cost of simply being alive
    energy_from_food: float = 12.0
    reproduce_threshold: float = 35.0  # below this, trying to reproduce does nothing
    reproduce_cost: float = 18.0       # energy handed from parent to child

    # ------------------------- the crowding cost: the "Freedman" mechanism -------
    # When enabled, every nearby creature drains a little energy each tick. This is the
    # stand-in for unavoidable social interaction being costly in itself, as opposed to
    # density merely being high.
    crowding_cost_enabled: bool = False
    crowding_energy_cost: float = 0.12  # per neighbour, per tick

    # -------------------------------------------------------------- the evolution
    mutation_std: float = 0.12
    action_temperature: float = 1.0

    def variant(self, name: str, **changes) -> "SimConfig":
        """A copy of this config with a few numbers changed."""
        return replace(self, name=name, **changes)


DEFAULT = SimConfig()
