"""The treatment arm: every creature with its own network.

Identical to the control in every other respect. Same pen, same reward, same pretrained
starting behaviour, same seeds. The founders are all copies of the same trained policy, so
the two arms begin with exactly the same behaviour and differ only in whether creatures are
allowed to drift apart afterwards.

What this makes visible is a population coming apart rather than an average moving, which
is what Calhoun actually described and what a shared policy structurally cannot show.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.distributions import Categorical  # noqa: E402

from agent import Agent, draw_lifespan  # noqa: E402
from brain import ACTIONS, Brain  # noqa: E402
from config import UNIVERSE_25  # noqa: E402
from individual import Population  # noqa: E402
from rl import OFFSPRING_REWARD, SURVIVAL_REWARD, PolicyNet, _build_world  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run(cfg, policy, seed, update_every=200, probe_every=500, lr=0.003, gamma=0.99):
    rng = random.Random(seed)
    world = _build_world(cfg, rng)
    if cfg.n_feeders:
        world.place_feeders(cfg.n_feeders)
        world.food.clear()
        world.scatter_food(cfg.n_food, spread=cfg.feeder_spread)
    if cfg.nests_enabled:
        world.scatter_nests(cfg.n_nests, on_perimeter=cfg.nests_on_perimeter)

    pop = Population(capacity=cfg.max_population)
    optimiser = torch.optim.Adam(pop.parameters(), lr=lr)

    world.agents = []
    for _ in range(cfg.n_initial_agents):
        slot = pop.founder(policy)
        x, y = world.random_square()
        a = Agent(x, y, cfg.energy_start, Brain.random(rng), lifespan=draw_lifespan(cfg, rng))
        a.slot = slot
        if cfg.ageing_enabled:
            a.age = rng.randrange(0, max(1, cfg.maturity_age * 2))
        world.agents.append(a)

    trajectories: dict = {}
    history = []

    for tick in range(cfg.n_ticks):
        world.rebuild_occupancy(cfg.crowding_radius)
        movers = [a for a in world.agents if a.alive and not a.is_dependent and a.slot is not None]
        pups_in_care = [(a, a.parent) for a in world.agents
                        if a.alive and a.is_dependent and a.parent is not None]

        chosen = {}
        if movers:
            slots = [a.slot for a in movers]
            senses = torch.tensor(
                np.array([a.sense(world, cfg, neighbours=a.neighbours) for a in movers]),
                dtype=torch.float32,
            )
            dist = Categorical(logits=pop.action_scores(slots, senses))
            picks = dist.sample()
            log_probs = dist.log_prob(picks)
            for i, a in enumerate(movers):
                chosen[a] = ACTIONS[picks[i].item()]
                steps, rewards = trajectories.setdefault(a.slot, ([], []))
                steps.append(log_probs[i])
                rewards.append(0.0)

        births = 0
        newborns = []
        for a in movers:
            child = a.act(world, cfg, rng, action=chosen.get(a))
            if child is None:
                continue
            slot = pop.inherit(a.slot) if len(world.agents) + len(newborns) < cfg.max_population else None
            if slot is None:
                if child.nest is not None:
                    world.release_nest(*child.nest)
                    child.nest = None
                a.refund_birth(cfg)
                continue
            pop.forget_slot(optimiser, slot)     # no momentum from whoever held it before
            trajectories.pop(slot, None)         # nor any leftover experience
            child.slot = slot
            newborns.append(child)
            births += 1

        for a in world.agents:
            if a.alive and a.is_dependent:
                a.act(world, cfg, rng)

        for a in movers:
            if a.alive and a.slot in trajectories and trajectories[a.slot][1]:
                trajectories[a.slot][1][-1] += SURVIVAL_REWARD
        for pup, parent in pups_in_care:
            if (pup.alive and not pup.is_dependent and parent.slot in trajectories
                    and trajectories[parent.slot][1]):
                trajectories[parent.slot][1][-1] += OFFSPRING_REWARD

        survivors = []
        toll = {"starvation": 0, "old age": 0, "neglect": 0}
        for a in world.agents:
            if a.alive:
                survivors.append(a)
                continue
            if a.nest is not None:
                world.release_nest(*a.nest)
                a.nest = None
            if a.parent is not None:
                a._leave_the_parent()
            if a.cause_of_death in toll:
                toll[a.cause_of_death] += 1
            if a.slot is not None:
                pop.release(a.slot)
                trajectories.pop(a.slot, None)
                a.slot = None
        deaths = len(world.agents) - len(survivors)
        world.agents = survivors + newborns
        world.respawn_step()

        record = {"tick": tick, "population": world.population, "births": births,
                  "deaths": deaths, "died_neglected": toll["neglect"],
                  "died_starving": toll["starvation"], "died_old": toll["old age"],
                  "nests_free": len(world.nests) - len(world.occupied_nests)}
        measured = [a.neighbours for a in world.agents if a.neighbours is not None]
        record["mean_neighbours"] = (sum(measured) / len(measured)) if measured else 0.0
        n = world.population
        patch = (2 * cfg.crowding_radius + 1) ** 2
        expected = (n - 1) * patch / (world.width * world.height) if n > 1 else 0.0
        record["clustering"] = (record["mean_neighbours"] / expected) if expected > 0 else 0.0

        if (tick + 1) % update_every == 0:
            pop.learn(optimiser, trajectories, gamma=gamma)
            trajectories = {}

        if probe_every and tick % probe_every == 0:
            live = [a.slot for a in world.agents if a.slot is not None and not a.is_dependent]
            record.update(pop.differentiation(live))

        history.append(record)
        if world.population == 0:
            break

    return world, pop, history


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")]
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 6000

    for seed in seeds:
        policy = PolicyNet()
        policy.load_state_dict(torch.load(ROOT / "results" / f"pen_policy_s{seed}.pt"))
        cfg = UNIVERSE_25.variant(f"individual_s{seed}", n_ticks=ticks, seed=seed)
        world, pop, history = run(cfg, policy, seed)

        out = ROOT / "results" / f"campaign_individual_s{seed}.json"
        json.dump(history, open(out, "w"))

        peak = max(h["population"] for h in history)
        tot = lambda k: sum(h[k] for h in history)
        probed = [h for h in history if "pup_mean" in h]
        print(f"seed {seed}: peak {peak}, final {history[-1]['population']}, "
              f"neglect {tot('died_neglected')}, starved {tot('died_starving')}", flush=True)
        print(f"   pup-seeking {probed[0]['pup_mean']:.3f} -> {probed[-1]['pup_mean']:.3f} | "
              f"spread {probed[0]['pup_sd']:.4f} -> {probed[-1]['pup_sd']:.4f} | "
              f"disengaged {probed[0]['disengaged']:.2f} -> {probed[-1]['disengaged']:.2f} | "
              f"weights {probed[0]['weight_spread']:.3f} -> {probed[-1]['weight_spread']:.3f}",
              flush=True)
