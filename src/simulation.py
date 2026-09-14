"""The loop that runs one whole experiment.

This is where evolution actually happens. There is no training step anywhere. We simply
let agents live, and the ones that reach the reproduction threshold leave mutated copies
of themselves behind. Over many ticks the population fills up with descendants of whoever
managed to eat reliably.

One run is one condition. The same loop runs every condition in the ablation study, and
only the config differs between them. Nothing about the logic changes, which is what makes
the comparison between conditions a fair one.
"""

from __future__ import annotations

import random
import time

from agent import Agent
from brain import Brain
from config import SimConfig
from world import World


def _new_population(world: World, cfg: SimConfig, rng: random.Random) -> list[Agent]:
    """The first generation. Random positions, random brains."""
    agents = []
    for _ in range(cfg.n_initial_agents):
        x, y = world.random_square()
        agents.append(Agent(x, y, cfg.energy_start, Brain.random(rng)))
    return agents


def run(cfg: SimConfig, progress_every: int = 0) -> dict:
    """Run one condition from start to finish.

    Returns a dict with the config name, the per tick history, and some summary numbers.

    The rng is seeded from the config, so the same config always produces exactly the same
    run. That matters for a report. Anyone can reproduce the figures.
    """
    rng = random.Random(cfg.seed)

    world = World(
        width=cfg.width,
        height=cfg.height,
        n_food=cfg.n_food,
        food_unlimited=cfg.food_unlimited,
        food_respawn_rate=cfg.food_respawn_rate,
        wrap_edges=cfg.wrap_edges,
        rng=rng,
    )
    world.agents = _new_population(world, cfg, rng)

    history: list[dict] = []
    started = time.time()
    extinct_at = None

    for tick in range(cfg.n_ticks):
        # Build the index of who is standing where, once, before anybody moves. Every
        # agent this tick therefore senses the same snapshot of the world, taken at the
        # start of the tick. This is simultaneous update: what an agent does depends on how
        # things were when the tick began, not on how far down the list it happens to sit.
        # The alternative, rebuilding after every single move, would make an agent's senses
        # depend on its position in the loop, which is not a property of the agent at all.
        world.rebuild_occupancy()

        births = 0
        newborns: list[Agent] = []

        # Agents act in a shuffled order each tick. Without this, the agent at index zero
        # would always eat first, which is a small but real advantage that has nothing to
        # do with its brain.
        acting = list(world.agents)
        rng.shuffle(acting)

        for agent in acting:
            if not agent.alive:
                continue
            child = agent.act(world, cfg, rng)
            if child is not None:
                if len(world.agents) + len(newborns) < cfg.max_population:
                    newborns.append(child)
                    births += 1
                else:
                    # At the cap the birth cannot happen, so give the parent back what it
                    # spent. The cap protects the program from running away, and it should
                    # not act as an invisible tax on breeding.
                    agent.refund_birth(cfg)

        survivors = [a for a in world.agents if a.alive]
        deaths = len(world.agents) - len(survivors)
        world.agents = survivors + newborns

        world.respawn_step()

        history.append(_snapshot(world, cfg, tick, births, deaths))

        if progress_every and tick % progress_every == 0:
            print(
                f"  tick {tick:5d}  population {world.population:4d}  "
                f"food {len(world.food):4d}"
            )

        if world.population == 0:
            extinct_at = tick
            break

    return {
        "name": cfg.name,
        "config": cfg,
        "history": history,
        "extinct_at": extinct_at,
        "final_population": world.population,
        "seconds": round(time.time() - started, 2),
        # The surviving agents are kept so their evolved brains can be examined or
        # benchmarked after the run. This is how we check that evolution did anything.
        "survivors": list(world.agents),
    }


def _snapshot(world: World, cfg: SimConfig, tick: int, births: int, deaths: int) -> dict:
    """One row of the results table. Everything the plots and the report will need.

    mean_neighbours is the measured crowding actually experienced by the agents. It is not
    the same as density. Density counts agents per square across the whole grid. This
    counts how many others each agent actually has close by, which is higher whenever the
    population clumps together instead of spreading out.
    """
    agents = world.agents
    n = len(agents)
    if n == 0:
        return {
            "tick": tick, "population": 0, "births": births, "deaths": deaths,
            "mean_energy": 0.0, "mean_age": 0.0, "mean_neighbours": 0.0,
            "density": 0.0, "food": len(world.food),
        }

    # Crowding is read back from what each agent actually measured when it acted, rather
    # than recounted here. That is both cheaper and more meaningful: it is the crowding the
    # agents really experienced and responded to. Newborns have not acted yet, so they are
    # left out of the average instead of counting as having no neighbours.
    measured = [a.neighbours for a in agents if a.neighbours is not None]

    return {
        "tick": tick,
        "population": n,
        "births": births,
        "deaths": deaths,
        "mean_energy": sum(a.energy for a in agents) / n,
        "mean_age": sum(a.age for a in agents) / n,
        "mean_neighbours": (sum(measured) / len(measured)) if measured else 0.0,
        "density": world.density,
        "food": len(world.food),
    }
