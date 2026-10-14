"""Independent matrix/accounting/statistical verifier; optional frozen replays."""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.neural_policy_evaluation import load_policies, trial, state_hash


def verify(data, analysis, replay=False):
    data,analysis=Path(data),Path(analysis)
    manifest=json.loads((data/'manifest.json').read_text());saved=json.loads((analysis/'analysis.json').read_text())
    for n,h in json.loads((data/'INTEGRITY.json').read_text()).items():assert hashlib.sha256((data/n).read_bytes()).hexdigest()==h
    for n,h in saved['input_sha256'].items():assert hashlib.sha256((data/n).read_bytes()).hexdigest()==h
    assert hashlib.sha256((ROOT/'experiments/neural_policy_evaluation.py').read_bytes()).hexdigest()==manifest['runner_sha256']
    assert hashlib.sha256((ROOT/'experiments/final_rewards.py').read_bytes()).hexdigest()==manifest['original_evaluator_sha256']
    policies,hashes=load_policies(range(10));assert hashes==manifest['checkpoint_sha256']
    assert {str(s):state_hash(policies[(s,'initial')]) for s in range(10)}==manifest['initial_parameter_hashes']
    cells=['initial','survival_normal','offspring_normal','survival_masked','offspring_masked']
    records=json.loads((data/'records.json').read_text());indexed={}
    noops=audits=0
    for r in records:
        key=(r['training_seed'],r['world_seed'],r['energy'],r['cell']);assert key not in indexed;indexed[key]=r
        o=r['outcome'];assert o['seed']==key[1] and o['initial_energy']==key[2]
        assert o['starting_population']==2 and o['births']==0
        assert 2-o['deaths']==o['final_population']
        assert o['deaths']==int(o['parent_cause_of_death'] is not None)+int(o['pup_cause_of_death'] is not None)
        assert o['parent_alive']==(o['parent_cause_of_death'] is None)
        assert 0 <= o['attended_ticks'] <= o['care_ticks'] <= o['elapsed_ticks'] <= 60
        assert o['parent_outcome']==('dead' if not o['parent_alive'] else 'survived_full_trial' if o['elapsed_ticks']==60 else 'alive_at_early_stop')
        assert (o['pup_cause_of_death'] is None)==o['independent']
        obj=r['cell'].split('_')[0];masked=r['cell'].endswith('masked')
        if obj!='initial' and not masked:
            old=json.loads((ROOT/f'results/final_reward_v1/{obj}_s{key[0]}_evaluation.json').read_text())
            assert o==next(t for t in old['assigned_pup_trials'] if t['seed']==key[1] and t['initial_energy']==key[2]);noops+=1
        for a in r['input_audits']:
            assert a['raw_input'][:9]==a['policy_input'][:9]
            assert a['policy_input'][9:]==([0.,0.,0.] if masked else a['raw_input'][9:])
            assert np.isclose(sum(a['normal_probabilities']),1,atol=1e-6)
            assert np.isclose(sum(a['applied_probabilities']),1,atol=1e-6)
            assert a['argmax_changed']==(np.argmax(a['normal_probabilities'])!=np.argmax(a['applied_probabilities']))
            if not masked:assert a['normal_probabilities']==a['applied_probabilities'] and a['normal_to_applied_kl']==0
            audits+=1
    assert len(indexed)==manifest['expected_trials']==2000
    assert set(indexed)==set(itertools.product(range(10),range(100000,100020),(10,20),cells))
    for s,w,e in itertools.product(range(10),range(100000,100020),(10,20)):
        assert len({json.dumps(indexed[(s,w,e,c)]['outcome']['initial_conditions'],sort_keys=True) for c in cells})==1
    summary=json.loads((data/'summary.json').read_text())
    for group in summary['groups']:
        rows=[r['outcome'] for r in records if r['cell']==group['cell'] and (group['energy'] is None or r['energy']==group['energy'])]
        assert len(rows)==group['trials']
        assert sum(o['independent'] for o in rows)==group['successes']
        for field,target in [('attended_ticks','attended_ticks'),('care_ticks','care_ticks'),('parent_meals','parent_meals')]:assert sum(o[field] for o in rows)==group[target]
        for field,prefix in [('parent_cause_of_death','parent'),('pup_cause_of_death','pup')]:
            assert {cause:sum(o[field]==cause for o in rows) for cause in ('starvation','old age','neglect')}==group[prefix+'_deaths_by_cause']
        assert group['parent_deaths']==sum(not o['parent_alive'] for o in rows)
        assert group['posthumous_independence']==sum(o['independent'] and not o['parent_alive'] for o in rows)
    pairs={'survival_training':('survival_normal','initial'),'offspring_training':('offspring_normal','initial'),
           'survival_masking':('survival_masked','survival_normal'),'offspring_masking':('offspring_masked','offspring_normal')}
    checked=[]
    for name,(a,b) in pairs.items():
        x=np.array([sum(int(indexed[(s,w,e,a)]['outcome']['independent'])-int(indexed[(s,w,e,b)]['outcome']['independent']) for w,e in itertools.product(range(100000,100020),(10,20)))/40 for s in range(10)])
        ci=np.percentile(np.random.default_rng(20261014).choice(x,(100000,10)).mean(axis=1),[2.5,97.5])
        p=sum(abs(sum(sign*v for sign,v in zip(pattern,x))/10)>=abs(x.mean())-1e-12 for pattern in itertools.product((-1,1),repeat=10))/1024
        t=next(t for t in saved['contrasts'] if t['contrast']==name)
        assert np.allclose(x,t['seed_effects'],atol=1e-12,rtol=0)
        assert np.allclose(ci,t['ci95'],atol=1e-12,rtol=0) and abs(x.mean()-t['mean'])<1e-12 and p==t['exact_p']
        checked.append({'contrast':name,'mean':float(x.mean()),'ci95':ci.tolist(),'exact_p':p})
    running=0
    for i,t in enumerate(sorted(saved['contrasts'],key=lambda t:t['exact_p'])):
        running=max(running,(4-i)*t['exact_p']);assert min(1,running)==t['holm_p']
    torch.set_num_threads(1);replays=0
    if replay:
        for s,e,c in itertools.product(range(10),(10,20),cells):
            policy=policies[(s,c.split('_')[0])];before=state_hash(policy)
            actual=json.loads(json.dumps(trial(policy,100019,e,c.endswith('masked'))))
            old=indexed[(s,100019,e,c)]
            assert actual=={k:old[k] for k in ('outcome','input_audits')}
            assert state_hash(policy)==before;replays+=1
    return {'trials':2000,'unique_initial_networks':10,'trained_networks':20,'normal_historical_replays':noops,
            'input_audits':audits,'exact_replays':replays,'contrasts':checked,'all_checks_passed':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'results/neural_policy_v1');p.add_argument('--analysis',type=Path,default=ROOT/'results/neural_policy_v1_analysis');p.add_argument('--replay',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); result=verify(a.data,a.analysis,a.replay)
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
