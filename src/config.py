"""Every parameter of the simulation, kept in one place.

This matters for this project in particular. The three experimental conditions differ only
in these numbers. No logic differs between them at all. That is what makes the comparison
between conditions a fair one.
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

    # ------------------------- the crowding cost, which is Freedman's mechanism ---
    # When enabled, each nearby agent drains a little energy per tick. This represents
    # unavoidable social interaction being costly in itself, as opposed to density simply
    # being high.
    crowding_cost_enabled: bool = False
    crowding_energy_cost: float = 0.12  # per neighbour, per tick

    # -------------------------------------------------------------- the evolution
    mutation_std: float = 0.12         # standard deviation of the mutation noise
    action_temperature: float = 1.0    # softmax temperature when sampling an action

    def variant(self, name: str, **changes) -> "SimConfig":
        """A copy of this config with some parameters changed."""
        return replace(self, name=name, **changes)


DEFAULT = SimConfig()
