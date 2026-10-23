"""Versioned supplied-family task. Agent/World dynamics and rewards are reused.

The intervention is the initial-state distribution and a reproduction threshold
of infinity. Independence is NOT a terminal condition. No shaping is supplied.
All random streams are local; final-test seeds cannot be collected by this API.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import random

import torch
from torch import nn

from agent import Agent, draw_lifespan
from brain import Brain
from config import UNIVERSE_25
from world import World
from .data import RewardConfig
from .environment import collect
from .evaluation import state_hash

TASK_VERSION = 'supplied-care-v3.0'
TRAIN_START, TRAIN_SIZE = 2_100_000, 100_000
VALIDATION_SEEDS = tuple(range(3_000_300, 3_000_304))
# A declaration of the sealed namespace; never instantiate these worlds here.
SEALED_START, SEALED_END = 4_000_000, 5_000_000


@dataclass(frozen=True)
class CareCondition:
    food_distance: int
    parent_energy: float
    parent_distance: int

    def __post_init__(self):
        if self.food_distance not in (4, 6, 8) or self.parent_energy not in (6., 12., 20.) or self.parent_distance not in (0, 4):
            raise ValueError('condition outside versioned task support')


CONDITIONS = tuple(CareCondition(d, e, p) for d in (4, 6, 8)
                   for e in (6., 12., 20.) for p in (0, 4))


def checked_seed(seed, split):
    valid = (TRAIN_START <= seed < TRAIN_START + TRAIN_SIZE if split == 'train'
             else seed in VALIDATION_SEEDS if split == 'validation'
             else 9_000_000 <= seed < 9_010_000 if split == 'technical' else False)
    if not valid or SEALED_START <= seed < SEALED_END:
        raise ValueError('seed outside authorized split; final-test panel is sealed')


class CareTask:
    def __init__(self, followup_ticks=40):
        if followup_ticks < 20 or followup_ticks > 100:
            raise ValueError('20..100 post-independence ticks required')
        self.followup_ticks = followup_ticks
        # Body parameters (energy, maturation, care, ageing) are inherited intact.
        self.cfg = UNIVERSE_25.variant(TASK_VERSION, n_ticks=60+followup_ticks,
            n_initial_agents=2, max_population=2, reproduce_threshold=float('inf'))

    def initialize(self, seed, condition, split='train'):
        checked_seed(seed, split)
        rng = random.Random(seed)
        size = rng.choice((24, 32))
        wrap = rng.choice((False, True))
        spread = rng.choice((0, 1))
        cfg = self.cfg.variant(TASK_VERSION, width=size, height=size,
            wrap_edges=wrap, n_food=1 if spread == 0 else 3,
            n_feeders=1, feeder_spread=spread, n_nests=1)
        world = World(size, size, 0, food_unlimited=cfg.food_unlimited,
            food_respawn_rate=cfg.food_respawn_rate, wrap_edges=wrap, rng=rng,
            feeder_spread=spread)
        # Margin keeps the constructed challenge off walls, even at distance 8.
        x, y = rng.randrange(10, size-9), rng.randrange(10, size-9)
        dx, dy = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        fx = x + dx*(condition.food_distance+spread)
        fy = y + dy*(condition.food_distance+spread)
        world.feeders.add((fx, fy))
        # Include a nearest-edge food item and random additional patch items.
        edge = (x+dx*condition.food_distance, y+dy*condition.food_distance)
        world.food.add(edge)
        world.scatter_food(cfg.n_food-1)
        world.nests.add((x, y))
        if not world.claim_nest(x, y):
            raise RuntimeError('supplied-pup nest was not free')
        parent = Agent(x+dx*condition.parent_distance, y+dy*condition.parent_distance,
                       condition.parent_energy, Brain.random(rng), draw_lifespan(cfg, rng))
        parent.age = cfg.maturity_age
        pup = Agent(x, y, cfg.reproduce_cost, Brain.random(rng), draw_lifespan(cfg, rng))
        pup.dependent_until = cfg.dependency_ticks
        pup.parent, pup.nest = parent, (x, y)
        parent.dependents = [pup]
        parent.children = 1  # supplied historical birth, not a new in-task birth
        world.agents = [parent, pup]
        return cfg, world, rng

    def collect(self, policy, seed, condition=None, *, objective='offspring',
                split='train', max_ticks=None):
        checked_seed(seed, split)
        condition = condition or CONDITIONS[(seed-TRAIN_START) % len(CONDITIONS)]
        cfg, world, rng = self.initialize(seed, condition, split)
        parent, pup = world.agents
        state = dict(independence_tick=None, parent_alive_at_independence=False,
            posthumous_independence=False, parent_death_tick=None, pup_death_tick=None,
            parent_at_risk_ticks=0, pup_at_risk_ticks=0, parent_feeds=0, pup_feeds=0)
        initial = dict(condition=asdict(condition), config=dict(width=cfg.width,
            height=cfg.height, wrap_edges=cfg.wrap_edges, feeder_spread=cfg.feeder_spread),
            parent_position=[parent.x,parent.y], pup_position=[pup.x,pup.y],
            food=sorted(world.food), feeders=sorted(world.feeders), nests=sorted(world.nests),
            parent_energy=parent.energy, pup_energy=pup.energy, parent_age=parent.age,
            pup_age=pup.age, parent_children=parent.children,
            parent_lifespan=parent.lifespan, pup_lifespan=pup.lifespan)
        # Observe actual successful take_food calls. This delegates once to the
        # original bound method, changes no result and draws no random numbers.
        original_take_food = world.take_food
        actors, food_calls = [], []
        def take_food(x,y):
            result = original_take_food(x,y)
            index = len(food_calls)
            if index >= len(actors):
                raise RuntimeError('unexpected food-consumption call')
            actor = actors[index]
            food_calls.append(result)
            if result:
                state['parent_feeds' if actor is parent else 'pup_feeds'] += 1
            return result
        world.take_food = take_food
        def observe(w,tick,stage):
            if stage == 'before':
                actors[:] = [a for a in w.agents if a.alive and a.age+1 >= a.dependent_until]
                food_calls.clear()
                state['parent_at_risk_ticks'] += int(parent.alive)
                state['pup_at_risk_ticks'] += int(pup.alive)
            else:
                if not parent.alive and state['parent_death_tick'] is None:
                    state['parent_death_tick'] = tick+1
                if not pup.alive and state['pup_death_tick'] is None:
                    state['pup_death_tick'] = tick+1
                if pup.alive and not pup.is_dependent and state['independence_tick'] is None:
                    state['independence_tick'] = tick+1
                    state['parent_alive_at_independence'] = parent.alive
                    state['posthumous_independence'] = not parent.alive
        try:
            episode = collect(cfg,policy,seed,RewardConfig(objective), action_seed=seed,
                max_ticks=max_ticks,initial=lambda c,s:(world,rng),observer=observe)
        finally:
            world.take_food = original_take_food
        if episode.metrics['births'] or episode.metrics['cap_rejections']:
            raise RuntimeError('supplied-pup task must contain no additional births')
        episode.metrics.update(state, task_version=TASK_VERSION, split=split,
            condition=asdict(condition), initial_conditions=initial,
            independent=state['independence_tick'] is not None,
            parent_alive=parent.alive, pup_alive=pup.alive,
            parent_death_cause=parent.cause_of_death,pup_death_cause=pup.cause_of_death,
            parent_adult_decisions=sum(t.agent==0 for t in episode.transitions),
            pup_adult_decisions=sum(t.agent==1 for t in episode.transitions),
            outcome_followup_ticks=cfg.n_ticks if not episode.truncated else episode.metrics['elapsed_ticks'],
            absorbing_ticks=cfg.n_ticks-episode.metrics['elapsed_ticks'] if not episode.truncated else 0)
        return episode


def summarize_care(records):
    n=len(records)
    totals={k:sum(r[k] for r in records) for k in ('independent',
        'parent_alive_at_independence','posthumous_independence','parent_alive',
        'pup_alive','care_ticks','attended_ticks','parent_feeds','pup_feeds',
        'parent_at_risk_ticks','pup_at_risk_ticks','adult_decisions')}
    totals.update(trials=n,independence_fraction=totals['independent']/n if n else None,
        parent_alive_independence_fraction=totals['parent_alive_at_independence']/n if n else None,
        parent_survival_fraction=totals['parent_alive']/n if n else None,
        attendance_fraction=totals['attended_ticks']/totals['care_ticks'] if totals['care_ticks'] else None,
        fed_parent_trial_count=sum(r['parent_feeds']>0 for r in records),
        parent_feeds_per_at_risk_tick=totals['parent_feeds']/totals['parent_at_risk_ticks'] if totals['parent_at_risk_ticks'] else None)
    return totals


def evaluate_care(task,policy,*,seeds=VALIDATION_SEEDS,conditions=CONDITIONS,split='validation'):
    if not seeds or not conditions or len(set(seeds))!=len(seeds) or len(set(conditions))!=len(conditions):
        raise ValueError('nonempty unique panel required')
    before,mode=state_hash(policy),policy.training
    try:
        policy.eval()
        with torch.no_grad():
            episodes=[task.collect(policy,s,c,split=split) for s in seeds for c in conditions]
    finally:
        policy.train(mode)
    if state_hash(policy)!=before:
        raise RuntimeError('frozen validation changed weights')
    records=[dict(world_seed=e.world_seed,action_seed=e.action_seed,initial_hash=e.initial_hash,
        policy_hash=before,**e.metrics) for e in episodes]
    return dict(task_version=TASK_VERSION,policy_class=type(policy).__name__,policy_hash=before,
        evaluation_reward_basis='canonical offspring (.01 survival + 1 independence), for every method',
        summary=summarize_care(records),
        by_food_distance={str(d):summarize_care([r for r in records if r['condition']['food_distance']==d]) for d in (4,6,8)},
        by_energy={str(e):summarize_care([r for r in records if r['condition']['parent_energy']==e]) for e in (6.,12.,20.)},
        trials=records)


def direction(dx,dy):
    if dx==dy==0:return 0
    return (3 if dx>0 else 4) if abs(dx)>=abs(dy) else (2 if dy>0 else 1)


class CareControl(nn.Module):
    """Observation-only, deterministic descriptive controls; no hidden world state.

    fixed_care returns to pup and waits, abandonment always walks away. The
    forage_care comparator prioritizes visible food below 40% energy, then care;
    at 15% energy it explores away from the pup when food is invisible. These
    choices apply ONLY to controls, never to neural policies or rewards.
    """
    def __init__(self,behavior='fixed_care'):
        super().__init__()
        if behavior not in ('fixed_care','abandonment','forage_care'):
            raise ValueError('unknown control')
        self.behavior=behavior
        self.register_buffer('_identity',torch.tensor(('fixed_care','abandonment','forage_care').index(behavior)))

    def forward(self,x):
        scores=torch.full((*x.shape[:-1],6),-1000.)
        for row,obs in zip(scores.reshape(-1,6),x.reshape(-1,12)):
            dx,dy=map(float,obs[9:11])
            if self.behavior=='abandonment':
                action=direction(-dx,-dy) if dx or dy else 3
            elif self.behavior=='forage_care' and obs[3]<.4 and obs[2]>0:
                action=direction(float(obs[0]),float(obs[1]))
            elif self.behavior=='forage_care' and obs[3]<.15:
                action=direction(-dx,-dy) if dx or dy else 3
            elif obs[11]>0:
                action=direction(dx,dy)
            elif self.behavior=='forage_care' and obs[2]>0:
                action=direction(float(obs[0]),float(obs[1]))
            else:action=0
            row[action]=0.
        return scores


def panel_fingerprint():
    return hashlib.sha256(json.dumps(dict(version=TASK_VERSION,seeds=VALIDATION_SEEDS,
        conditions=[asdict(c) for c in CONDITIONS]),sort_keys=True).encode()).hexdigest()
