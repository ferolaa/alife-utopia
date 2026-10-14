"""Frozen-policy care stress assay, not a natural colony or Calhoun replication.

Zero/eight immobile, resource-neutral background bodies enter the ordinary occupancy
index and policy crowding sense. Only the parent and pup act. Background bodies have
no physiology, eat nothing, claim no nests and cannot block movement. Their placement
uses a separate RNG and cannot change initial geometry or the food respawn RNG.

The existing care formula is unchanged (its tick-start count includes a nearby
parent, whereas attendance uses the post-movement parent position). Consequently,
even zero-background disruption can block care if a parent moves from radius two
to distance three. Background-only and total crowding are reported separately.
Crowding energy cost, intrusion, developmental damage and further reproduction are
disabled in this evaluation only. Ordinary metabolism, feeding, movement and neglect
remain unchanged. All trials last 60 ticks, including after early pup death, to give
parent survival a common follow-up. Only focal bodies enter mortality accounting.

Crowding remains visible to the policy: density can change actions intentionally.
Within-density disruption off/on isolates the intervention on care effectiveness;
across-density contrasts also include the policy's response to perceived crowding.
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from agent import Agent, draw_lifespan
from brain import ACTIONS, Brain
from config import UNIVERSE_25
from rl import PolicyNet, _build_world

HIGH_BACKGROUND = 8
HORIZON = UNIVERSE_25.dependency_ticks


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_hash(policy):
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                 for p in policy.parameters())).hexdigest()


@dataclass(frozen=True)
class BackgroundBody:
    x: int
    y: int
    # Minimal occupancy-only object; deliberately has no act(), energy or nest.


def setup(seed, energy, n_background, disruption):
    if n_background not in (0, HIGH_BACKGROUND):
        raise ValueError('Prespecified background counts are 0 and 8')
    if energy not in (10, 20):
        raise ValueError('Prespecified parent energies are 10 and 20')
    cfg = UNIVERSE_25.variant('controlled_care', n_ticks=HORIZON,
                             crowding_cost_enabled=False,
                             crowding_blocks_care=disruption,
                             nest_intrusion_enabled=False,
                             developmental_damage_enabled=False,
                             reproduce_threshold=float('inf'))
    rng = random.Random(seed)
    world = _build_world(cfg, rng)
    candidates = []
    for x, y in sorted(world.nests):
        distance = min(max(abs(x-fx), abs(y-fy)) for fx, fy in world.food)
        if not 4 <= distance <= 8:
            continue
        sites = [(sx, sy) for sx in range(max(0,x-2), min(cfg.width,x+3))
                 for sy in range(max(0,y-2), min(cfg.height,y+3))
                 if (sx,sy) not in world.nests and (sx,sy) not in world.food]
        # Apply this constraint in BOTH density conditions, before selecting a trial.
        if len(sites) < HIGH_BACKGROUND:
            continue
        for dx, dy in ((4,0),(-4,0),(0,4),(0,-4)):
            px, py = x+dx, y+dy
            if 0 <= px < cfg.width and 0 <= py < cfg.height and any(
                    max(abs(px-fx), abs(py-fy)) <= cfg.vision for fx,fy in world.food):
                candidates.append((x,y,px,py,sites))
    if not candidates:
        raise ValueError(f'No matched care geometry fits seed {seed}; no fallback or retuning')
    x,y,px,py,sites = rng.choice(candidates)
    parent = Agent(px,py,energy,Brain.random(rng),lifespan=draw_lifespan(cfg,rng))
    parent.age = cfg.maturity_age
    pup = Agent(x,y,cfg.reproduce_cost,Brain.random(rng),lifespan=draw_lifespan(cfg,rng))
    pup.parent, pup.dependent_until, pup.nest = parent, cfg.dependency_ticks, (x,y)
    parent.dependents = [pup]
    assert world.claim_nest(x,y)
    # Does not consume the world/physiology RNG; low density is an empty subset.
    positions = random.Random(seed + 2718281828).sample(sites,HIGH_BACKGROUND)
    background = [BackgroundBody(*pos) for pos in positions[:n_background]]
    world.agents = [parent,pup,*background]
    initial = {'parent': {'position':[px,py], 'energy':energy, 'age':parent.age,
                           'lifespan':parent.lifespan, 'dependent_until':parent.dependent_until},
               'pup': {'position':[x,y], 'energy':pup.energy, 'age':pup.age,
                       'lifespan':pup.lifespan, 'dependent_until':pup.dependent_until,
                       'neglect_ticks':pup.neglect_ticks, 'parent_assigned':True},
               'feeders':sorted(world.feeders), 'food':sorted(world.food),
               'nests':sorted(world.nests), 'claimed_nests':sorted(world.occupied_nests)}
    return cfg,rng,world,parent,pup,background,initial


def trial(policy, seed, energy, n_background, disruption):
    cfg,rng,world,parent,pup,background,initial = setup(seed,energy,n_background,disruption)
    counts = Counter()
    history = []
    before_weights = policy_hash(policy)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.random.default_generator.manual_seed(seed)
        for tick in range(1,HORIZON+1):
            world.rebuild_occupancy(cfg.crowding_radius)
            movers = [a for a in (parent,pup) if a.alive and a.age+1 >= a.dependent_until]
            parent_crowd = world.count_neighbours(parent.x,parent.y,cfg.crowding_radius,
                                                   exclude=parent) if parent.alive else None
            choices = []
            if movers:
                senses = torch.tensor(np.array([a.sense(world,cfg) for a in movers]),
                                      dtype=torch.float32)
                choices = torch.distributions.Categorical(logits=policy(senses)).sample().tolist()
            actions = {}
            parent_energy_before = parent.energy
            for a,choice in zip(movers,choices):
                action = ACTIONS[choice]
                actions['parent' if a is parent else 'pup'] = action
                meals = [0]
                take_food = world.take_food
                def take(x,y):
                    eaten = take_food(x,y)
                    meals[0] += eaten
                    return eaten
                world.take_food = take
                try:
                    child = a.act(world,cfg,rng,action=action)
                finally:
                    world.take_food = take_food
                assert child is None, 'Extra reproduction is excluded from assigned-pup assay'
                counts['parent_meals' if a is parent else 'pup_meals'] += meals[0]
                if a is parent:
                    counts['parent_action_ticks'] += 1
                    counts['parent_crowd_sum'] += parent_crowd
            care = None
            if pup.alive and pup.is_dependent:
                crowd = world.count_neighbours(pup.x,pup.y,cfg.crowding_radius,exclude=pup)
                distance = max(abs(parent.x-pup.x),abs(parent.y-pup.y))
                baseline = parent.alive and not parent.is_dependent and distance <= cfg.care_radius
                effective = cfg.care_radius/(1+cfg.care_radius_decay*crowd)
                blocked = baseline and distance > effective
                old_neglect = pup.neglect_suffered
                pup.act(world,cfg,rng)
                attended = pup.neglect_suffered == old_neglect
                assert attended == (baseline and (not disruption or not blocked))
                counts['care_ticks'] += 1
                counts['attended_ticks'] += attended
                counts['pup_crowd_sum'] += crowd
                counts['third_party_crowd_sum'] += len(background)
                counts['crowd_blocked_ticks'] += disruption and blocked
                counts['counterfactual_blockable_ticks'] += blocked
                care = {'crowd':crowd,'third_party_crowd':len(background),
                        'baseline_attendance':baseline,'attended':attended,
                        'crowd_blocked':disruption and blocked,'effective_radius':effective}
            focal = [a for a in (parent,pup) if a.alive]
            for a in (parent,pup):
                if not a.alive:
                    if a.nest is not None:
                        world.release_nest(*a.nest)
                        a.nest = None
                    if a.parent is not None:
                        a._leave_the_parent()
            world.agents = [*focal,*background]
            world.respawn_step()
            assert len(world.food)==cfg.n_food
            assert world.occupied_nests <= {(pup.x,pup.y)}
            history.append({'tick':tick,'focal_population':len(focal),
                            'total_occupancy_bodies':len(world.agents),
                            'food_count':len(world.food),'food_hash':digest(sorted(world.food)),
                            'parent_alive':parent.alive,'pup_alive':pup.alive,
                            'parent_position':[parent.x,parent.y], 'parent_age':parent.age,
                            'parent_energy_before':parent_energy_before,
                            'parent_energy_after':parent.energy,'parent_crowd':parent_crowd,
                            'actions':actions,'care':care})
    assert policy_hash(policy)==before_weights
    mortality = {cause:sum(not a.alive and a.cause_of_death==cause for a in (parent,pup))
                 for cause in ('starvation','old age','neglect')}
    deaths = sum(mortality.values())
    assert 2-deaths == sum(a.alive for a in (parent,pup))
    assert len(world.agents)==2+len(background)-deaths
    return {'world_seed':seed,'initial_energy':energy,'background_count':n_background,
            'disruption':disruption,'initial_conditions':initial,'initial_hash':digest(initial),
            'background_positions':[[b.x,b.y] for b in background],
            'independent':pup.alive and not pup.is_dependent,'parent_alive':parent.alive,
            'parent_cause_of_death':parent.cause_of_death,'pup_cause_of_death':pup.cause_of_death,
            'elapsed_ticks':HORIZON,'starting_population':2,'births':0,'deaths':deaths,
            'final_population':2-deaths,'background_population':len(background),
            'final_total_occupancy_bodies':len(world.agents),'deaths_by_cause':mortality,
            'counts':dict(counts),'history':history}


def summarize(records):
    groups = []
    for objective in ('survival','offspring'):
        for energy in (10,20):
            for n in (0,HIGH_BACKGROUND):
                for disruption in (False,True):
                    group = [r for r in records if r['objective']==objective and
                             r['initial_energy']==energy and r['background_count']==n and
                             r['disruption']==disruption]
                    counts, mortality = Counter(),Counter()
                    for r in group:
                        counts.update(r['counts']); mortality.update(r['deaths_by_cause'])
                    groups.append({'objective':objective,'initial_energy':energy,
                                   'background_count':n,'disruption':disruption,'trials':len(group),
                                   'independence_successes':sum(r['independent'] for r in group),
                                   'parent_survivors':sum(r['parent_alive'] for r in group),
                                   'counts':dict(counts),'deaths_by_cause':dict(mortality)})
    pairs = []
    for seed in sorted({r['training_seed'] for r in records}):
        for n in (0,HIGH_BACKGROUND):
            for disruption in (False,True):
                rates = {}
                for objective in ('survival','offspring'):
                    group = [r for r in records if r['training_seed']==seed and
                             r['objective']==objective and r['background_count']==n and
                             r['disruption']==disruption]
                    rates[objective] = sum(r['independent'] for r in group)/len(group)
                pairs.append({'training_seed':seed,'background_count':n,'disruption':disruption,
                              **rates,'difference':rates['offspring']-rates['survival']})
    return {'groups':groups,'training_pair_rates':pairs,
            'inference':'Pilot descriptive only; worlds and energies are not independent training replicates'}


def execute(args):
    out = Path(args.out).resolve()
    checkpoints = (ROOT/'results/final_reward_v1').resolve()
    if out==ROOT/'results' or out==checkpoints or checkpoints in out.parents:
        raise ValueError('Output must be a new directory outside existing final results')
    training_seeds = [int(s) for s in args.seeds.split(',')]
    world_seeds = [int(s) for s in args.world_seeds.split(',')]
    if len(set(training_seeds))!=len(training_seeds) or len(set(world_seeds))!=len(world_seeds):
        raise ValueError('Duplicate seeds would create duplicate trials')
    original_manifest = json.loads((checkpoints/'manifest.json').read_text())
    used = set(original_manifest['evaluation_seeds']) | {
        s*1000+i for s in original_manifest['training_seeds']
        for i in range(original_manifest['iterations'])}
    if used & set(world_seeds):
        raise ValueError('Use fresh evaluation seeds, disjoint from final training/evaluation')
    files = {f'{objective}_s{s}.pt':file_hash(checkpoints/f'{objective}_s{s}.pt')
             for s in training_seeds for objective in ('survival','offspring')}
    out.mkdir(parents=True,exist_ok=False)
    manifest = {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'runner_sha256':file_hash(Path(__file__)), 'training_seeds':training_seeds,
                'world_seeds':world_seeds,'energies':[10,20],'background_counts':[0,HIGH_BACKGROUND],
                'horizon':HORIZON,'checkpoint_sha256':files,'torch':torch.__version__,
                'intervention':__doc__,'crowding_energy_penalty':False,'retraining':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    started = time.perf_counter()
    records = []
    for s in training_seeds:
        for objective in ('survival','offspring'):
            policy = PolicyNet(seed=s)
            policy.load_state_dict(torch.load(checkpoints/f'{objective}_s{s}.pt',weights_only=True))
            policy.eval()
            subset = []
            for world_seed in world_seeds:
                for energy in (10,20):
                    for n in (0,HIGH_BACKGROUND):
                        for disruption in (False,True):
                            record = trial(policy,world_seed,energy,n,disruption)
                            record.update(training_seed=s,objective=objective)
                            subset.append(record)
            records.extend(subset)
            (out/f'{objective}_s{s}.json').write_text(json.dumps(subset,indent=2))
            print(objective,s,'trials',len(subset),flush=True)
    for world_seed in world_seeds:
        for energy in (10,20):
            group = [r for r in records if r['world_seed']==world_seed and r['initial_energy']==energy]
            assert len({r['initial_hash'] for r in group})==1
            assert all(r['counts']['third_party_crowd_sum']==r['background_count']*r['counts']['care_ticks']
                       for r in group)
    assert all(file_hash(checkpoints/name)==h for name,h in files.items())
    summary = summarize(records)
    summary.update(total_seconds=time.perf_counter()-started,completed_trials=len(records),
                   expected_trials=len(training_seeds)*2*len(world_seeds)*2*2*2,
                   matching_verified=True,checkpoint_integrity_verified=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--seeds',default='0,1')
    parser.add_argument('--world-seeds',default='300000,300001,300002')
    torch.set_num_threads(1)
    execute(parser.parse_args())
