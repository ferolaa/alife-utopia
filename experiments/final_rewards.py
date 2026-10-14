"""Isolated reward comparison: paired training and frozen held-out evaluation.

This runner does not change training or core dynamics. Assigned-pup challenges are
explicit evaluation interventions: one parent/pup, no additional reproduction,
parent initially four squares from the pup, initial energy 20 or 10.
"""
import argparse
import hashlib
import itertools
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from agent import Agent, draw_lifespan
from brain import ACTIONS, Brain
from config import UNIVERSE_25
from represent import probe
from rl import OFFSPRING_REWARD, SURVIVAL_REWARD, PolicyNet, _build_world, run_episode, train


def state_hash(policy):
    return hashlib.sha256(b''.join(p.detach().numpy().tobytes()
                                 for p in policy.parameters())).hexdigest()


def natural_episode(cfg, policy, seed):
    """Observe the existing episode loop; no learning or intervention on dynamics."""
    births, grown, seen, founders = {}, set(), set(), set()
    populations = []
    clock = [0]
    exposures = [0, 0]
    fingerprint = []
    original_rebuild = __import__('world').World.rebuild_occupancy
    original_respawn = __import__('world').World.respawn_step
    original_care = Agent._spend_tick_as_pup
    original_act = Agent.act

    def rebuild(world, radius=None):
        if not fingerprint:
            seen.update(world.agents)
            founders.update(world.agents)
            initial = [sorted(world.feeders), sorted(world.food), sorted(world.nests),
                       [(a.x, a.y, a.age, a.energy, a.lifespan) for a in world.agents]]
            fingerprint.append(hashlib.sha256(json.dumps(initial).encode()).hexdigest())
        clock[0] += 1
        return original_rebuild(world, radius)

    def respawn(world):
        for a in world.agents:
            if a not in seen:
                seen.add(a)
                births[a] = clock[0]
        populations.append(world.population)
        return original_respawn(world)

    def care(a, world, cfg):
        before = a.neglect_suffered
        original_care(a, world, cfg)
        exposures[0] += 1
        exposures[1] += a.neglect_suffered == before

    def act(a, world, cfg, rng, action=None):
        was_dependent = a.is_dependent
        child = original_act(a, world, cfg, rng, action=action)
        if was_dependent and not a.is_dependent and a.alive:
            grown.add(a)
        return child

    with patch('world.World.rebuild_occupancy', rebuild), patch('world.World.respawn_step', respawn), \
            patch.object(Agent, '_spend_tick_as_pup', care), patch.object(Agent, 'act', act), \
            torch.no_grad():
        _, _, stats = run_episode(cfg, policy, False, seed)
    eligible = [a for a, tick in births.items() if tick + cfg.dependency_ticks <= cfg.n_ticks]
    successful = [a for a in eligible if a in grown]
    assert stats['births'] == len(births)
    causes = ('starvation', 'old age', 'neglect')
    dead = {a for a in seen if not a.alive}
    assert all(a.cause_of_death in causes for a in dead)
    mortality = {cause: sum(a.cause_of_death == cause for a in dead) for cause in causes}
    founder_mortality = {cause: sum(a.cause_of_death == cause for a in dead & founders)
                         for cause in causes}
    offspring_mortality = {cause: sum(a.cause_of_death == cause for a in dead - founders)
                           for cause in causes}
    assert len(founders) + len(births) - len(dead) == stats['final_population']
    return {'seed': seed, **stats, 'initial_conditions_hash': fingerprint[0],
            'eligible_births': len(eligible), 'independent_offspring': len(successful),
            'independence_fraction': len(successful)/len(eligible) if eligible else None,
            'excluded_late_births': len(births)-len(eligible),
            'starting_population': len(founders), 'deaths': len(dead),
            'deaths_by_cause': mortality, 'founder_deaths_by_cause': founder_mortality,
            'offspring_deaths_by_cause': offspring_mortality,
            'neglect_deaths': mortality['neglect'], 'elapsed_ticks': clock[0],
            'population_history': populations, 'peak_population': max([len(founders), *populations]),
            'care_ticks': exposures[0], 'attended_ticks': exposures[1],
            'attendance_fraction': exposures[1]/exposures[0] if exposures[0] else None}


def assigned_pup_trial(policy, seed, energy):
    """Frozen policy raising one supplied pup, using unchanged Agent.act dynamics."""
    cfg = UNIVERSE_25.variant('assigned_pup_evaluation', n_ticks=60,
                             reproduce_threshold=float('inf'))
    rng = random.Random(seed)
    world = _build_world(cfg, rng)
    candidates = []
    for x, y in sorted(world.nests):
        # Existing food is outside care radius, but reachable without extreme travel.
        distance = min(max(abs(x-fx), abs(y-fy)) for fx, fy in world.food)
        if not 4 <= distance <= 8:
            continue
        for dx, dy in ((4, 0), (-4, 0), (0, 4), (0, -4)):
            px, py = x+dx, y+dy
            if 0 <= px < cfg.width and 0 <= py < cfg.height and any(
                    max(abs(px-fx), abs(py-fy)) <= cfg.vision for fx, fy in world.food):
                candidates.append((x, y, px, py))
    if not candidates:
        raise ValueError(f'No prespecified care challenge fits world seed {seed}')
    x, y, px, py = rng.choice(candidates)
    parent = Agent(px, py, energy, Brain.random(rng), lifespan=draw_lifespan(cfg, rng))
    parent.age = cfg.maturity_age
    pup = Agent(x, y, cfg.reproduce_cost, Brain.random(rng), lifespan=draw_lifespan(cfg, rng))
    pup.parent = parent
    pup.dependent_until = cfg.dependency_ticks
    pup.nest = (x, y)
    parent.dependents = [pup]
    assert world.claim_nest(x, y)
    world.agents = [parent, pup]
    initial = {'nest': [x, y], 'parent_position': [px, py], 'parent_energy': energy,
               'feeders': sorted(world.feeders), 'food': sorted(world.food), 'nests': sorted(world.nests)}
    attended, care_ticks, meals = 0, 0, [0]
    active = [None]
    original_take_food = world.take_food

    def take_food(x, y):
        eaten = original_take_food(x, y)
        if eaten and active[0] is parent:
            meals[0] += 1
        return eaten

    world.take_food = take_food
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.random.default_generator.manual_seed(seed)
        for elapsed in range(1, cfg.dependency_ticks + 1):
            world.rebuild_occupancy(cfg.crowding_radius)
            movers = [a for a in world.agents if a.alive and a.age+1 >= a.dependent_until]
            if movers:
                senses = torch.tensor(np.array([a.sense(world, cfg) for a in movers]), dtype=torch.float32)
                choices = torch.distributions.Categorical(logits=policy(senses)).sample()
                for a, choice in zip(movers, choices):
                    active[0] = a
                    a.act(world, cfg, rng, action=ACTIONS[choice.item()])
                    active[0] = None
            if pup.alive and pup.is_dependent:
                before = pup.neglect_suffered
                pup.act(world, cfg, rng)
                if pup.is_dependent:
                    care_ticks += 1
                    attended += pup.neglect_suffered == before
            survivors = []
            for a in world.agents:
                if a.alive:
                    survivors.append(a)
                else:
                    if a.nest is not None:
                        world.release_nest(*a.nest)
                        a.nest = None
                    if a.parent is not None:
                        a._leave_the_parent()
            world.agents = survivors
            world.respawn_step()
            if not pup.alive:
                break
    return {'seed': seed, 'initial_energy': energy, 'initial_conditions': initial,
            'independent': pup.alive and not pup.is_dependent, 'parent_alive': parent.alive,
            'pup_cause_of_death': pup.cause_of_death, 'care_ticks': care_ticks,
            'attended_ticks': attended, 'neglect_suffered': pup.neglect_suffered,
            'parent_meals': meals[0], 'parent_cause_of_death': parent.cause_of_death,
            'parent_outcome': ('dead' if not parent.alive else
                               'survived_full_trial' if elapsed == cfg.dependency_ticks else
                               'alive_at_early_stop'),
            'elapsed_ticks': elapsed, 'planned_ticks': cfg.dependency_ticks,
            'starting_population': 2, 'births': 0,
            'deaths': int(not parent.alive) + int(not pup.alive),
            'final_population': len(world.agents)}


def paired_analysis(pairs):
    """Inference uses independent training pairs, not individual evaluation trials."""
    differences = np.array([p['difference'] for p in pairs], dtype=float)
    result = {'n_training_pairs': len(pairs), 'mean_difference': float(differences.mean()),
              'paired_bootstrap_95_percent_interval': None, 'exact_sign_flip_p': None}
    if len(pairs) > 1:
        samples = np.random.default_rng(20261014).choice(
            differences, size=(10000, len(pairs)), replace=True).mean(axis=1)
        result['paired_bootstrap_95_percent_interval'] = np.quantile(samples, [.025,.975]).tolist()
        if len(pairs) <= 20:
            extreme = sum(abs(np.dot(signs, differences)/len(pairs)) >=
                          abs(differences.mean()) - 1e-12
                          for signs in itertools.product((-1,1), repeat=len(pairs)))
            result['exact_sign_flip_p'] = extreme / 2**len(pairs)
    return result


def outcome_summary(results):
    """Descriptive totals; inference remains at independent training-pair level."""
    summary = {}
    for objective in ('survival', 'offspring'):
        records = [r for r in results if r['objective'] == objective]
        natural = [e for r in records for e in r['natural_episodes']]
        trials = [t for r in records for t in r['assigned_pup_trials']]
        summary[objective] = {
            'natural_episodes': len(natural),
            'births': sum(e['births'] for e in natural),
            'eligible_births': sum(e['eligible_births'] for e in natural),
            'independent_offspring': sum(e['independent_offspring'] for e in natural),
            'deaths_by_cause': {c: sum(e['deaths_by_cause'][c] for e in natural)
                               for c in ('starvation', 'old age', 'neglect')},
            'founder_deaths_by_cause': {c: sum(e['founder_deaths_by_cause'][c] for e in natural)
                                      for c in ('starvation', 'old age', 'neglect')},
            'offspring_deaths_by_cause': {c: sum(e['offspring_deaths_by_cause'][c] for e in natural)
                                        for c in ('starvation', 'old age', 'neglect')},
            'care_ticks': sum(e['care_ticks'] for e in natural),
            'attended_ticks': sum(e['attended_ticks'] for e in natural),
            'mean_final_population': float(np.mean([e['final_population'] for e in natural])),
            'extinct_episodes': sum(e['final_population'] == 0 for e in natural),
            'assigned_pup_trials': len(trials),
            'assigned_pup_successes': sum(t['independent'] for t in trials),
            'assigned_parent_meals': sum(t['parent_meals'] for t in trials),
            'assigned_parent_outcomes': {label: sum(t['parent_outcome'] == label for t in trials)
                                        for label in ('dead', 'survived_full_trial', 'alive_at_early_stop')},
            'assigned_parent_deaths_by_cause': {c: sum(t['parent_cause_of_death'] == c for t in trials)
                                              for c in ('starvation', 'old age', 'neglect')},
            'assigned_care_ticks': sum(t['care_ticks'] for t in trials),
            'assigned_attended_ticks': sum(t['attended_ticks'] for t in trials),
        }
    return summary


def execute(args):
    out = Path(args.out).resolve()
    # Never allow overwriting or writing directly into the legacy artifact folder.
    if out == (ROOT/'results').resolve():
        raise ValueError('Choose a new output subdirectory, not legacy results/')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg = UNIVERSE_25.variant('final_reward_comparison', n_ticks=args.ticks,
                            n_initial_agents=40, max_population=120)
    training_seeds = [int(s) for s in args.seeds.split(',')]
    evaluation_seeds = [int(s) for s in args.eval_seeds.split(',')]
    training_world_seeds = {s*1000+i for s in training_seeds for i in range(args.iterations)}
    if training_world_seeds & set(evaluation_seeds):
        raise ValueError('Evaluation world seeds overlap training world seeds')
    manifest = {'commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'working_tree_status': subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
                'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                'training_config': asdict(cfg), 'iterations': args.iterations,
                'training_seeds': training_seeds, 'evaluation_seeds': evaluation_seeds,
                'evaluation_ticks': args.eval_ticks, 'lr': .01, 'gamma': .99,
                'survival_reward': SURVIVAL_REWARD, 'offspring_reward': OFFSPRING_REWARD,
                'training_world_seed_rule': 'training_seed * 1000 + iteration',
                'challenge_energies': [20,10], 'challenge_initial_distance':4,
                'limitations': ['dead individual trajectories unresolved; individual mode excluded',
                                'continuous windows truncate delayed credit; continuous learning excluded',
                                'episodic horizon truncation and parent-death credit remain',
                                'parent attendance post-action, crowding tick-start remains',
                                'offspring survival measured at independence; later adult death does not erase success']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    all_results=[]
    started=time.perf_counter()
    for seed in training_seeds:
        initial_hash=state_hash(PolicyNet(seed=seed))
        for objective in ('survival','offspring'):
            begin=time.perf_counter()
            policy,history=train(cfg,objective=='offspring',iterations=args.iterations,seed=seed,
                                 log_every=0,probe_every=0)
            policy.eval()
            trained_hash=state_hash(policy)
            torch.save(policy.state_dict(),out/f'{objective}_s{seed}.pt')
            (out/f'{objective}_s{seed}_training.json').write_text(json.dumps(history,indent=2))
            natural=[natural_episode(cfg.variant('heldout',n_ticks=args.eval_ticks),policy,s)
                     for s in evaluation_seeds]
            challenges=[assigned_pup_trial(policy,s,e) for s in evaluation_seeds for e in (20,10)]
            assert state_hash(policy)==trained_hash, 'evaluation changed policy weights'
            result={'objective':objective,'training_seed':seed,'initial_policy_hash':initial_hash,
                    'trained_policy_hash':trained_hash,'natural_episodes':natural,
                    'assigned_pup_trials':challenges,'behavior_probes':probe(policy.to_brain(),trials=300),
                    'primary_independence_fraction':sum(t['independent'] for t in challenges)/len(challenges),
                    'elapsed_seconds':time.perf_counter()-begin}
            all_results.append(result)
            (out/f'{objective}_s{seed}_evaluation.json').write_text(json.dumps(result,indent=2))
            print(objective,seed,'care_success',result['primary_independence_fraction'],
                  'seconds',round(result['elapsed_seconds'],2),flush=True)
    pairs=[]
    for seed in training_seeds:
        a,b=[r for r in all_results if r['training_seed']==seed]
        assert a['initial_policy_hash']==b['initial_policy_hash']
        assert [x['initial_conditions_hash'] for x in a['natural_episodes']]==[
            x['initial_conditions_hash'] for x in b['natural_episodes']]
        assert [x['initial_conditions'] for x in a['assigned_pup_trials']]==[
            x['initial_conditions'] for x in b['assigned_pup_trials']]
        pairs.append({'seed':seed,'survival':a['primary_independence_fraction'],
                      'offspring':b['primary_independence_fraction'],
                      'difference':b['primary_independence_fraction']-a['primary_independence_fraction']})
    summary={'paired_primary_endpoint':pairs, 'paired_analysis':paired_analysis(pairs),
             'outcome_summary':outcome_summary(all_results),
             'total_seconds':time.perf_counter()-started}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--seeds',default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--iterations',type=int,default=120)
    parser.add_argument('--ticks',type=int,default=500)
    parser.add_argument('--eval-seeds',default=','.join(str(s) for s in range(100000,100020)))
    parser.add_argument('--eval-ticks',type=int,default=1000)
    execute(parser.parse_args())
