"""Data collection over unchanged Agent/World dynamics; no optimizer or autograd graph.

Order and pre-action observations intentionally match legacy run_episode. These
observations precede age/cost/maturation processing; Agent.last_senses differs.
"""
import hashlib
import json
import random

import numpy as np
import torch

from agent import Agent, draw_lifespan
from brain import ACTIONS, Brain
from rl import _build_world
from .data import Episode, Event, RewardConfig, Transition


def initialize(cfg, seed):
    rng = random.Random(seed)
    world = _build_world(cfg, rng)
    world.agents = []
    # Match legacy founder RNG ordering exactly (brains first, then placement).
    for brain in [Brain.random(rng) for _ in range(cfg.n_initial_agents)]:
        x, y = world.random_square()
        a = Agent(x, y, cfg.energy_start, brain, lifespan=draw_lifespan(cfg, rng))
        if cfg.ageing_enabled:
            a.age = rng.randrange(0, max(1, cfg.fertility_end_age))
        world.agents.append(a)
    return world, rng


def fingerprint(world):
    initial = [sorted(world.feeders), sorted(world.food), sorted(world.nests),
               [(a.x, a.y, a.age, a.energy, a.lifespan) for a in world.agents]]
    return hashlib.sha256(json.dumps(initial).encode()).hexdigest()


def collect(cfg, policy, world_seed, rewards=RewardConfig(), *, action_seed=None,
            max_ticks=None, initial=None, observer=None):
    """A finite-horizon world task. max_ticks is an external truncation, not terminal.

    Environment interaction count = sampled adult decisions, including fatal ones.
    World ticks, adult exposure and pup-care exposure are recorded separately.
    IDs and birth ownership survive parent death and child detachment.
    `initial` is an injectable world initializer for mathematical scenario tests.
    """
    if cfg.n_ticks <= 0 or cfg.n_initial_agents < 0:
        raise ValueError('positive horizon and nonnegative initial population required')
    if max_ticks is not None and max_ticks <= 0:
        raise ValueError('max_ticks must be positive')
    world, rng = (initial or initialize)(cfg, world_seed)
    ids = {a: i for i, a in enumerate(world.agents)}
    birth_ticks, ownership = {}, {}
    for a in world.agents:
        if a.parent is not None:
            ownership[a] = ids[a.parent]
    action_seed = world_seed if action_seed is None else action_seed
    generator = torch.Generator().manual_seed(action_seed)
    episode = Episode(world_seed, action_seed, fingerprint(world), cfg.n_ticks)
    births = deaths = independent = attended = care_ticks = adult_survived = 0
    population = []
    successful = set()
    action_counts = {action: 0 for action in ACTIONS}
    reproduction_failures = cap_rejections = 0
    causes = {'starvation': 0, 'old age': 0, 'neglect': 0}
    limit = min(cfg.n_ticks, max_ticks) if max_ticks is not None else cfg.n_ticks
    elapsed = 0
    for tick in range(limit):
        if not world.agents:
            break
        elapsed = tick + 1
        if observer is not None:
            observer(world, tick, "before")
        world.rebuild_occupancy(cfg.crowding_radius)
        movers = [a for a in world.agents if a.alive and a.age + 1 >= a.dependent_until]
        pups = [(a, ownership.get(a)) for a in world.agents if a.alive and a.is_dependent]
        records = {}
        if movers:
            observations = np.array([a.sense(world, cfg) for a in movers])
            with torch.no_grad():
                logits = policy(torch.tensor(observations, dtype=torch.float32))
                if logits.shape != (len(movers), 6) or not torch.isfinite(logits).all():
                    raise ValueError('policy must produce finite [agents, 6] logits')
                logp = logits.log_softmax(-1)
                actions = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
            for i, a in enumerate(movers):
                record = Transition(tick, ids[a], tuple(observations[i]), int(actions[i]),
                                    float(logp[i, actions[i]]))
                action_counts[ACTIONS[record.action]] += 1
                records[a] = record
                episode.transitions.append(record)
        newborns = []
        for a in movers:
            child = a.act(world, cfg, rng, action=ACTIONS[records[a].action])
            if records[a].action == 5 and child is None:
                reproduction_failures += 1
            if child is not None:
                if len(world.agents) + len(newborns) < cfg.max_population:
                    ids[child] = len(ids)
                    ownership[child] = ids[a]
                    birth_ticks[child] = tick + 1
                    newborns.append(child)
                    births += 1
                else:
                    cap_rejections += 1
                    if child.nest is not None:
                        world.release_nest(*child.nest)
                        child.nest = None
                    a.refund_birth(cfg)
        for a in world.agents:
            if a.alive and a.is_dependent:
                before = a.neglect_suffered
                a.act(world, cfg, rng)
                care_ticks += 1
                attended += a.neglect_suffered == before
        for a in movers:
            records[a].terminated = not a.alive
            if a.alive:
                adult_survived += 1
                episode.events.append(Event(tick, ids[a], rewards.survival, 'survival'))
        for pup, parent_id in pups:
            if pup.alive and not pup.is_dependent:
                independent += 1
                successful.add(ids[pup])
                if parent_id is not None and rewards.objective == 'offspring':
                    episode.events.append(Event(tick, parent_id, rewards.independence,
                                                'independence', ids[pup]))
        survivors = []
        for a in world.agents:
            if a.alive:
                survivors.append(a)
            else:
                deaths += 1
                causes[a.cause_of_death] += 1
                if a.nest is not None:
                    world.release_nest(*a.nest)
                    a.nest = None
                if a.parent is not None:
                    a._leave_the_parent()
        world.agents = survivors + newborns
        world.respawn_step()
        population.append(world.population)
        if observer is not None:
            observer(world, tick, "after")
    episode.truncated = bool(world.agents and limit < cfg.n_ticks and elapsed == limit)
    # Mark the final decision of each surviving actor on external truncation. A pup
    # may have no decisions yet. Finite task horizons are not external truncations.
    if episode.truncated:
        for a in world.agents:
            record = next((t for t in reversed(episode.transitions) if t.agent == ids[a]), None)
            if record is not None:
                record.truncated = True
    eligible = sum(b + cfg.dependency_ticks <= cfg.n_ticks for b in birth_ticks.values())
    eligible_success = sum(ids[a] in successful for a, b in birth_ticks.items()
                           if b + cfg.dependency_ticks <= cfg.n_ticks)
    episode.metrics = dict(elapsed_ticks=elapsed, planned_ticks=cfg.n_ticks,
        starting_population=len(ids)-births, births=births, deaths=deaths,
        final_population=world.population, peak_population=max([len(ids)-births, *population]),
        population_history=population, deaths_by_cause=causes,
        adult_decisions=episode.decisions, surviving_adult_ticks=adult_survived,
        action_counts=action_counts, reproduction_failures=reproduction_failures,
        cap_rejections=cap_rejections,
        care_ticks=care_ticks, attended_ticks=attended,
        attendance_fraction=attended/care_ticks if care_ticks else None,
        independent_offspring=independent, eligible_births=eligible,
        excluded_late_births=births-eligible,
        eligible_independent_offspring=eligible_success,
        independence_fraction=eligible_success/eligible if eligible and not episode.truncated else None,
        total_reward=sum(e.amount for e in episode.events),
        reward_per_decision=sum(e.amount for e in episode.events)/episode.decisions
                            if episode.decisions else None,
        horizon_complete=not episode.truncated, extinct=not bool(world.agents))
    return episode
