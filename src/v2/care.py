"""Standardized frozen assigned-pup assay, with full 60-tick outcome follow-up.

The intervention supplies one mature parent and its pup, disables further births,
and places the parent four cardinal steps from a nest near visible food. All
Agent/World rules remain unchanged. Unlike historical assays, pup death does not
stop follow-up of a surviving parent. If both die, remaining ticks are absorbing.
"""
import hashlib
import json
import random

import torch

from agent import Agent, draw_lifespan
from brain import Brain
from rl import _build_world
from .data import RewardConfig
from .environment import collect
from .evaluation import state_hash


def care_initialization(energy):
    def initialize(cfg, seed):
        rng = random.Random(seed)
        world = _build_world(cfg, rng)
        candidates = []
        for x, y in sorted(world.nests):
            distance = min((max(abs(x-fx), abs(y-fy)) for fx, fy in world.food), default=float('inf'))
            if not 4 <= distance <= 8:
                continue
            for dx, dy in ((4, 0), (-4, 0), (0, 4), (0, -4)):
                px, py = x+dx, y+dy
                if (0 <= px < cfg.width and 0 <= py < cfg.height and any(
                    max(abs(px-fx), abs(py-fy)) <= cfg.vision for fx, fy in world.food)):
                    candidates.append((x, y, px, py))
        if not candidates:
            raise ValueError(f'No prespecified assigned-pup challenge fits seed {seed}')
        x, y, px, py = rng.choice(candidates)
        parent = Agent(px, py, energy, Brain.random(rng), lifespan=draw_lifespan(cfg, rng))
        parent.age = cfg.maturity_age
        pup = Agent(x, y, cfg.reproduce_cost, Brain.random(rng), lifespan=draw_lifespan(cfg, rng))
        pup.parent = parent
        pup.dependent_until = cfg.dependency_ticks
        pup.nest = (x, y)
        parent.dependents = [pup]
        if not world.claim_nest(x, y):
            raise RuntimeError('assigned nest was not free')
        world.agents = [parent, pup]
        return world, rng
    return initialize


def assigned_pup_trial(cfg, policy, seed, energy, *, initializer=None):
    if energy <= 0 or not cfg.parental_care_enabled or cfg.dependency_ticks <= 0:
        raise ValueError('assigned care requires positive energy and dependency')
    assay = cfg.variant('v2-assigned-pup', n_ticks=cfg.dependency_ticks,
                        n_initial_agents=2, reproduce_threshold=float('inf'))
    state = {}
    def initial(config, world_seed):
        world, rng = (initializer or care_initialization(energy))(config, world_seed)
        if len(world.agents) != 2:
            raise ValueError('assay initializer must supply exactly a parent and a pup')
        parent, pup = world.agents
        state.update(parent=parent, pup=pup, independence_tick=None, parent_death_tick=None,
                     pup_death_tick=None, posthumous=False, parent_at_risk=0, pup_at_risk=0)
        layout = [sorted(world.feeders), sorted(world.food), sorted(world.nests),
                  (parent.x, parent.y), (pup.x, pup.y)]
        state['layout_hash'] = hashlib.sha256(json.dumps(layout).encode()).hexdigest()
        state['initial_conditions'] = dict(parent_position=[parent.x, parent.y],
            nest=[pup.x, pup.y], parent_energy=parent.energy, parent_age=parent.age,
            parent_lifespan=parent.lifespan, pup_lifespan=pup.lifespan,
            feeders=sorted(world.feeders), food=sorted(world.food), nests=sorted(world.nests))
        return world, rng
    def observe(world, tick, stage):
        parent, pup = state['parent'], state['pup']
        if stage == 'before':
            state['parent_at_risk'] += int(parent.alive)
            state['pup_at_risk'] += int(pup.alive)
        else:
            if not parent.alive and state['parent_death_tick'] is None:
                state['parent_death_tick'] = tick + 1
            if not pup.alive and state['pup_death_tick'] is None:
                state['pup_death_tick'] = tick + 1
            if pup.alive and not pup.is_dependent and state['independence_tick'] is None:
                state['independence_tick'] = tick + 1
                state['posthumous'] = not parent.alive
    before, mode = state_hash(policy), policy.training
    try:
        policy.eval()
        with torch.no_grad():
            episode = collect(assay, policy, seed, RewardConfig('offspring'), action_seed=seed,
                              initial=initial, observer=observe)
    finally:
        policy.train(mode)
    if state_hash(policy) != before:
        raise RuntimeError('assigned-pup evaluation changed policy state')
    parent, pup = state['parent'], state['pup']
    if episode.metrics['births'] != 0 or episode.truncated:
        raise RuntimeError('assay must be complete and contain no new births')
    return dict(world_seed=seed, action_seed=seed, initial_energy=energy,
        initial_hash=episode.initial_hash, layout_hash=state['layout_hash'],
        initial_conditions=state['initial_conditions'], policy_hash=before,
        independent=state['independence_tick'] is not None,
        independence_tick=state['independence_tick'], posthumous_independence=state['posthumous'],
        parent_alive=parent.alive, parent_death_tick=state['parent_death_tick'],
        parent_cause_of_death=parent.cause_of_death,
        pup_alive=pup.alive, pup_death_tick=state['pup_death_tick'], pup_cause_of_death=pup.cause_of_death,
        planned_followup_ticks=assay.n_ticks, outcome_followup_ticks=assay.n_ticks,
        simulated_ticks=episode.metrics['elapsed_ticks'], absorbing_ticks=assay.n_ticks-episode.metrics['elapsed_ticks'],
        parent_at_risk_ticks=state['parent_at_risk'], pup_at_risk_ticks=state['pup_at_risk'],
        parent_adult_decisions=sum(t.agent == 0 for t in episode.transitions),
        adult_decisions=episode.decisions, care_ticks=episode.metrics['care_ticks'],
        attended_ticks=episode.metrics['attended_ticks'],
        attendance_fraction=episode.metrics['attendance_fraction'], births=0,
        action_counts=episode.metrics['action_counts'])


def summarize_assigned(records):
    """Trial success denominator stays separate from natural eligible births."""
    count = len(records)
    care = sum(r['care_ticks'] for r in records)
    attended = sum(r['attended_ticks'] for r in records)
    return dict(trials=count, independent=sum(r['independent'] for r in records),
        independence_fraction=sum(r['independent'] for r in records)/count if count else None,
        posthumous_independence=sum(r['posthumous_independence'] for r in records),
        care_ticks=care, attended_ticks=attended, attendance_fraction=attended/care if care else None,
        parent_deaths=sum(not r['parent_alive'] for r in records),
        parent_at_risk_ticks=sum(r['parent_at_risk_ticks'] for r in records),
        outcome_followup_ticks=sum(r['outcome_followup_ticks'] for r in records),
        adult_decisions=sum(r['adult_decisions'] for r in records))


def evaluate_assigned(cfg, policy, seeds, energies=(10, 20)):
    if len(set(energies)) != len(energies) or len(energies) != 2:
        raise ValueError('exactly two distinct energy conditions required')
    records = [assigned_pup_trial(cfg, policy, seed, energy)
               for seed in seeds.validation for energy in energies]
    return dict(summary=summarize_assigned(records),
                by_energy={str(e): summarize_assigned([r for r in records if r['initial_energy']==e])
                           for e in energies}, trials=records)
