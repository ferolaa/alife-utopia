"""Independently verify immutable Phase 1/2 data; optionally replay saved evaluations.

Reads plain JSON locally or byte-identical gzip records in the preservation archive.
Statistics are recomputed directly from trial outcomes, without calling runner analysis
functions. Confidence intervals are conditional on the fixed evaluation-world panel.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def raw(path):
    if path.is_file():
        return path.read_bytes()
    archive=ROOT/'results/controlled_care_v1_archive'/f'{path.name}.gz'
    return gzip.decompress(archive.read_bytes())


def load(path):
    return json.loads(raw(path))


def stats(values, repetitions):
    values=np.asarray(values,dtype=float)
    draws=np.random.default_rng(20261014).choice(values,size=(repetitions,10),replace=True)
    ci=np.percentile(draws.mean(axis=1),[2.5,97.5])
    extreme=0
    for signs in itertools.product((-1,1),repeat=10):
        statistic=sum(s*x for s,x in zip(signs,values))/10
        extreme+=abs(statistic)>=abs(values.mean())-1e-12
    return {'mean':float(values.mean()),'ci95':ci.tolist(),'p':extreme/1024}


def verify(replay=False):
    from rl import PolicyNet
    from experiments.final_rewards import state_hash
    phase1=ROOT/'results/final_reward_v1'
    phase2=ROOT/'results/controlled_care_v1'
    m1,m2=load(phase1/'manifest.json'),load(phase2/'manifest.json')
    assert m1['training_seeds']==m2['training_seeds']==list(range(10))
    assert m1['evaluation_seeds']==list(range(100000,100020))
    assert m2['world_seeds']==list(range(300000,300020))
    for name,m in (('final_rewards.py',m1),('controlled_care.py',m2)):
        assert sha((ROOT/'experiments'/name).read_bytes())==m['runner_sha256']
    training_worlds=[set(range(s*1000,s*1000+m1['iterations'])) for s in range(10)]
    assert sum(map(len,training_worlds))==len(set.union(*training_worlds))==1200
    assert not set.union(*training_worlds)&set(m1['evaluation_seeds']+m2['world_seeds'])
    records1=[]; policies={};initial_hashes=[]; trained_hashes=[]
    for s in range(10):
        initial_hashes.append(state_hash(PolicyNet(seed=s)))
        pair=[]
        for objective in ('survival','offspring'):
            record=load(phase1/f'{objective}_s{s}_evaluation.json');pair.append(record);records1.append(record)
            assert record['training_seed']==s and record['objective']==objective
            assert record['initial_policy_hash']==initial_hashes[-1]
            policy=PolicyNet(seed=s)
            policy.load_state_dict(torch.load(phase1/f'{objective}_s{s}.pt',weights_only=True));policy.eval()
            assert all(torch.isfinite(p).all() for p in policy.parameters())
            assert state_hash(policy)==record['trained_policy_hash']
            trained_hashes.append(state_hash(policy));policies[(s,objective)]=policy
            history=load(phase1/f'{objective}_s{s}_training.json')
            assert [h['iteration'] for h in history]==list(range(120))
            assert all(np.isfinite(h['loss']) and np.isfinite(h['mean_return']) for h in history)
            trials=record['assigned_pup_trials'];episodes=record['natural_episodes']
            assert len(trials)==40 and len(episodes)==20
            assert {(t['seed'],t['initial_energy']) for t in trials}==set(itertools.product(m1['evaluation_seeds'],(10,20)))
            assert [e['seed'] for e in episodes]==m1['evaluation_seeds']
            assert record['primary_independence_fraction']==sum(t['independent'] for t in trials)/40
            for row in [*trials,*episodes]:
                assert row['starting_population']+row['births']-row['deaths']==row['final_population']
            for e in episodes:
                assert e['eligible_births']+e['excluded_late_births']==e['births']
                assert e['independent_offspring']<=e['eligible_births']
                assert sum(e['deaths_by_cause'].values())==e['deaths']
                assert all(e['founder_deaths_by_cause'][c]+e['offspring_deaths_by_cause'][c]==n for c,n in e['deaths_by_cause'].items())
                assert len(e['population_history'])==e['elapsed_ticks']
                assert e['population_history'][-1]==e['final_population']
            for t in trials:
                assert t['elapsed_ticks']<=60 and t['care_ticks']<=59
                assert t['parent_alive']==(t['parent_cause_of_death'] is None)
        assert [t['initial_conditions'] for t in pair[0]['assigned_pup_trials']]==[t['initial_conditions'] for t in pair[1]['assigned_pup_trials']]
        assert [e['initial_conditions_hash'] for e in pair[0]['natural_episodes']]==[e['initial_conditions_hash'] for e in pair[1]['natural_episodes']]
    assert len(set(initial_hashes))==10 and len(set(trained_hashes))==20
    pairs=[next(r['primary_independence_fraction'] for r in records1 if r['training_seed']==s and r['objective']=='offspring')-
           next(r['primary_independence_fraction'] for r in records1 if r['training_seed']==s and r['objective']=='survival') for s in range(10)]
    p1=stats(pairs,10000);summary1=load(phase1/'summary.json')
    for objective in ('survival','offspring'):
        rows=[r for r in records1 if r['objective']==objective]
        episodes=[e for r in rows for e in r['natural_episodes']]
        trials=[t for r in rows for t in r['assigned_pup_trials']]
        saved=summary1['outcome_summary'][objective]
        for key in ('births','eligible_births','independent_offspring','care_ticks','attended_ticks'):
            assert saved[key]==sum(e[key] for e in episodes)
        for key in ('deaths_by_cause','founder_deaths_by_cause','offspring_deaths_by_cause'):
            assert saved[key]=={c:sum(e[key][c] for e in episodes) for c in ('starvation','old age','neglect')}
        assert saved['mean_final_population']==np.mean([e['final_population'] for e in episodes])
        assert saved['extinct_episodes']==sum(e['final_population']==0 for e in episodes)
        assert saved['assigned_pup_successes']==sum(t['independent'] for t in trials)
        assert saved['assigned_parent_meals']==sum(t['parent_meals'] for t in trials)
        assert saved['assigned_parent_outcomes']==dict(Counter(t['parent_outcome'] for t in trials))
        assert saved['assigned_parent_deaths_by_cause']=={c:sum(t['parent_cause_of_death']==c for t in trials) for c in ('starvation','old age','neglect')}
        assert saved['assigned_care_ticks']==sum(t['care_ticks'] for t in trials)
        assert saved['assigned_attended_ticks']==sum(t['attended_ticks'] for t in trials)
    assert np.allclose(p1['ci95'],summary1['paired_analysis']['paired_bootstrap_95_percent_interval'])
    assert p1['p']==summary1['paired_analysis']['exact_sign_flip_p']
    assert np.isclose(p1['mean'],summary1['paired_analysis']['mean_difference'])
    for s,p in enumerate(summary1['paired_primary_endpoint']):
        assert p['seed']==s and np.isclose(p['difference'],pairs[s])
    records2=[r for s in range(10) for objective in ('survival','offspring')
              for r in load(phase2/f'{objective}_s{s}.json')]
    keys={tuple(r[k] for k in ('training_seed','objective','world_seed','initial_energy','background_count','disruption')) for r in records2}
    assert len(records2)==len(keys)==3200
    for r in records2:
        assert len(r['history'])==r['elapsed_ticks']==60
        assert r['starting_population']==2 and r['births']==0
        assert 2-r['deaths']==r['final_population'] and sum(r['deaths_by_cause'].values())==r['deaths']
        assert r['final_total_occupancy_bodies']==r['final_population']+r['background_count']
        assert all(t['food_count']==60 for t in r['history'])
        assert r['initial_hash']==sha(json.dumps(r['initial_conditions'],sort_keys=True).encode())
        assert r['counts']['third_party_crowd_sum']==r['background_count']*r['counts']['care_ticks']
        care=[t['care'] for t in r['history'] if t['care'] is not None]
        for counter,field in (('care_ticks',None),('attended_ticks','attended'),('crowd_blocked_ticks','crowd_blocked')):
            assert r['counts'][counter]==(len(care) if field is None else sum(c[field] for c in care))
        assert r['counts']['pup_crowd_sum']==sum(c['crowd'] for c in care)
    for seed,energy in itertools.product(m2['world_seeds'],(10,20)):
        assert len({r['initial_hash'] for r in records2 if r['world_seed']==seed and r['initial_energy']==energy})==1
    analysis=load(ROOT/'results/controlled_care_v1_analysis/analysis.json'); rates={}
    for g in analysis['groups']:
        subset=[r for r in records2 if r['objective']==g['objective'] and r['background_count']==g['background'] and r['disruption']==g['disruption'] and (g['energy'] is None or r['initial_energy']==g['energy'])]
        assert len(subset)==g['trials'] and sum(r['independent'] for r in subset)==g['successes']
        counts,mort=Counter(),Counter()
        for r in subset:counts.update(r['counts']);mort.update(r['deaths_by_cause'])
        assert dict(counts)==g['counts'] and dict(mort)==g['deaths_by_cause']
        assert sum(r['independent'] and not r['parent_alive'] for r in subset)==g['success_after_parent_death']
        rates[(g['objective'],g['energy'],g['background'],g['disruption'])]=np.array([sum(r['independent'] for r in subset if r['training_seed']==s)/sum(r['training_seed']==s for r in subset) for s in range(10)])
        assert np.allclose(rates[(g['objective'],g['energy'],g['background'],g['disruption'])],g['seed_rates'])
    p2=[]
    for saved in analysis['paired_contrasts']:
        energy=saved['energy'];name=saved['contrast']
        if name.startswith('policy_advantage'):
            n=int(name.split('_')[3]);d=bool(int(name.split('_')[-1]));x=rates[('offspring',energy,n,d)]-rates[('survival',energy,n,d)]
        elif name.startswith('disruption_effect'):
            obj=name.split('_')[2];n=int(name.split('_')[-1]);x=rates[(obj,energy,n,True)]-rates[(obj,energy,n,False)]
        else:
            gap=lambda n,d:rates[('offspring',energy,n,d)]-rates[('survival',energy,n,d)]
            x=(gap(8,True)-gap(8,False))-(gap(0,True)-gap(0,False))
        independent=stats(x,100000)
        assert np.allclose(independent['ci95'],saved['ci95']) and np.isclose(independent['mean'],saved['mean'])
        assert independent['p']==saved['exact_two_sided_sign_flip_p']
        p2.append({'contrast':name,'energy':energy,**independent})
    for energy in (None,10,20):
        family=sorted([t for t in analysis['paired_contrasts'] if t['energy']==energy],key=lambda t:t['exact_two_sided_sign_flip_p']);running=0
        for i,t in enumerate(family):
            running=max(running,(9-i)*t['exact_two_sided_sign_flip_p'])
            assert np.isclose(min(1,running),t['holm_adjusted_p_within_nine_contrasts'])
    for name,h in analysis['output_sha256'].items():assert sha(raw(phase2/name))==h
    for name,h in m2['checkpoint_sha256'].items():assert sha((phase1/name).read_bytes())==h
    replayed=0
    if replay:
        from experiments.final_rewards import assigned_pup_trial,natural_episode
        from experiments.controlled_care import trial
        from config import SimConfig
        torch.set_num_threads(1)
        for r in records1:
            s,obj=r['training_seed'],r['objective'];p=policies[(s,obj)];before=state_hash(p)
            expected=next(t for t in r['assigned_pup_trials'] if t['seed']==100000 and t['initial_energy']==20)
            assert json.loads(json.dumps(assigned_pup_trial(p,100000,20)))==expected;replayed+=1
            expected2=next(t for t in records2 if t['training_seed']==s and t['objective']==obj and t['world_seed']==300019 and t['initial_energy']==20 and t['background_count']==8 and t['disruption'])
            actual=json.loads(json.dumps(trial(p,300019,20,8,True)))
            assert all(actual[k]==v for k,v in expected2.items() if k not in ('training_seed','objective'));replayed+=1
            if s==0:
                cfg=SimConfig(**m1['training_config']).variant('heldout',n_ticks=1000)
                assert natural_episode(cfg,p,100000)==r['natural_episodes'][0];replayed+=1
            assert state_hash(p)==before
    return {'phase1_primary':p1,'phase2_contrasts':p2,'verified_checkpoints':20,
            'unique_initializations':10,'unique_trained_weight_sets':20,'training_world_seeds':1200,
            'phase1_assigned_trials':800,'phase1_natural_episodes':400,'phase2_trials':3200,
            'evaluation_replays':replayed,'core_commit':m1['commit'],
            'limitations':'Intervals condition on the fixed world panel; n=10 training pairs. Phase1 parent follow-up is censored by pup death. Phase2 attendance denominator ends at pup death/independence. No biological or unconstrained population-collapse inference.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',action='store_true')
    parser.add_argument('--out',help='New JSON verification file; existing files refused')
    args=parser.parse_args();result=verify(args.replay)
    if args.out:
        with Path(args.out).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
