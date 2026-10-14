"""Independent raw-record/statistic verifier for full Phase 3; optional frozen replays."""
import argparse
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


def verify(data,analysis,replay=False):
    data,analysis=Path(data),Path(analysis)
    manifest=json.loads((data/'manifest.json').read_text());saved=json.loads((analysis/'analysis.json').read_text())
    assert manifest['training_seeds']==list(range(10))
    correction=None
    if manifest['expected_trajectories']!=1600:
        correction=json.loads((analysis/'METADATA_CORRECTION.json').read_text())
        assert correction['correct_expected_trajectories']==1600
        assert correction['original_expected_trajectories']==manifest['expected_trajectories']
        last=json.loads((ROOT/'results/final_reward_v1/offspring_s9_evaluation.json').read_text())['trained_policy_hash']
        assert manifest['expected_trajectories']==last
    assert manifest['unattended_age_interval']==[10,27] and manifest['assessment_energy']==20
    assert manifest['childhood_background']==8 and manifest['caregiver_distance']==2
    for key,base in (('adult_world_seeds',400000),('childhood_world_seeds',410000),('bridge_world_seeds',420000)):
        assert manifest[key]==list(range(base,base+20))
    expected={'manifest.json','summary.json'}|{f'{obj}_s{s}_w{400000+j}_c{c}_d{d}.json'
        for obj in ('survival','offspring') for s in range(10) for j in range(20) for c in (0,1) for d in (0,1)}
    assert {p.name for p in data.iterdir()}==expected
    records={}
    for name in sorted(expected-{'manifest.json','summary.json'}):
        r=json.loads((data/name).read_text())
        key=(r['training_seed'],r['objective'],r['world_index'],r['disrupted_childhood'],r['damage_enabled'])
        assert key not in records;records[key]=r
        child,bridge,adult=r['childhood'],r['bridge'],r['adult']
        assert r['original_cohort_size']==r['childhood_survivors']==r['mature_survivors']==r['adults_evaluated']==1
        assert len(child['history'])==len(bridge['history'])==len(adult['history'])==60
        missing=[h['tick'] for h in child['history'] if h['attended'] is False]
        assert missing==(list(range(10,28)) if r['disrupted_childhood'] else [])
        assert child['counts']['care_ticks']==59 and child['before_maturation']['neglect_ticks']==0
        assert child['final']['age']==60 and bridge['final']['age']==120
        assert child['final']['parent_attached'] is False
        assert all(h['crowd']==9 for h in child['history'][:59])
        assert all(h['caregiver']['alive'] for h in child['history'])
        assert child['caregiver_final']['energy']==45 and child['counts']['caregiver_meals']==0
        assert [h['age'] for h in bridge['history']]==list(range(61,121))
        assert bridge['final']['energy']>=2.75
        trait=.3 if r['disrupted_childhood'] and r['damage_enabled'] else 0
        assert child['final']['impairment']==bridge['final']['impairment']==adult['impairment']==adult['final_impairment']==trait
        assert adult['initial']['parent']['age']==120 and adult['initial']['parent']['energy']==20
        assert adult['elapsed_ticks']==60 and all(h['parent_alive'] for h in adult['history'])
        assert adult['parent_alive'] and adult['parent_cause_of_death'] is None
        assert adult['starting_population']+adult['births']-adult['deaths']==adult['final_population']
        assert sum(adult['deaths_by_cause'].values())==adult['deaths']
        assert adult['deaths_by_cause']['starvation']==adult['deaths_by_cause']['old age']==0
        assert adult['deaths_by_cause']['neglect']==int(not adult['independent'])
        assert adult['counts']['care_ticks']==sum(h['attended'] is not None for h in adult['history'])
        assert adult['counts']['attended_ticks']==sum(h['attended'] is True for h in adult['history'])
        for h in adult['history']:
            x=np.array(h['parent_policy_input']);reference=np.array(h['unattenuated_same_state_input'])
            assert np.array_equal(x[:9],reference[:9])
            assert np.allclose(x[9:],reference[9:]*(1-trait),rtol=0,atol=1e-12)
        assert all(h['food_count']==60 for stage in (child,bridge,adult) for h in stage['history'])
    assert len(records)==1600
    for s,obj,j in itertools.product(range(10),('survival','offspring'),range(20)):
        controls=[records[(s,obj,j,False,False)],records[(s,obj,j,False,True)],records[(s,obj,j,True,False)]]
        assert all(r['adult']==controls[0]['adult'] for r in controls)
        assert all(records[(s,obj,j,c,d)]['bridge']['history']==controls[0]['bridge']['history'] for c,d in itertools.product((False,True),repeat=2))
    for j in range(20):
        assert len({records[(s,obj,j,c,d)]['adult']['initial_hash'] for s,obj,c,d in itertools.product(range(10),('survival','offspring'),(False,True),(False,True))})==1
    effects={}
    for obj in ('survival','offspring'):
        # Integer success-count differences first, to avoid floating sign ambiguities.
        effects[obj]=np.array([sum((1 if c==d else -1)*int(records[(s,obj,j,c,d)]['adult']['independent'])
            for j in range(20) for c,d in itertools.product((False,True),repeat=2))/20
            for s in range(10)])
    effects['objective_interaction']=effects['offspring']-effects['survival']
    checked=[]
    for name,x in effects.items():
        ci=np.percentile(np.random.default_rng(20261014).choice(x,(100000,10),replace=True).mean(axis=1),[2.5,97.5])
        extremes=sum(abs(sum(a*b for a,b in zip(pattern,x))/10)>=abs(sum(x)/10)-1e-12
                     for pattern in itertools.product((-1,1),repeat=10))
        p=extremes/1024;t=next(t for t in saved['paired_contrasts'] if t['contrast']==name)
        assert np.allclose(ci,t['ci95'],rtol=0,atol=1e-12)
        assert np.isclose(sum(x)/10,t['mean'],rtol=0,atol=1e-12) and p==t['exact_two_sided_sign_flip_p']
        assert np.allclose(x,t['seed_effects'],rtol=0,atol=1e-12)
        checked.append({'contrast':name,'mean':float(sum(x)/10),'ci95':ci.tolist(),'p':p})
    running=0
    for i,t in enumerate(sorted(saved['paired_contrasts'],key=lambda t:t['exact_two_sided_sign_flip_p'])):
        running=max(running,(3-i)*t['exact_two_sided_sign_flip_p']);assert t['holm_adjusted_p']==min(1,running)
    for g in saved['groups']:
        selected=[r for key,r in records.items() if key[1]==g['objective'] and key[3]==g['disrupted_childhood'] and key[4]==g['damage_enabled']]
        assert len(selected)==g['assigned_cohort']==200
        assert sum(r['adult']['independent'] for r in selected)==g['independence_successes']
        counts,mort=Counter(),Counter()
        for r in selected:counts.update(r['adult']['counts']);mort.update(r['adult']['deaths_by_cause'])
        assert dict(counts)==g['counts'] and dict(mort)==g['deaths_by_cause']
    for name,h in saved['input_sha256'].items():assert hashlib.sha256((data/name).read_bytes()).hexdigest()==h
    source=ROOT/'experiments/developmental_care.py'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['runner_sha256']:
        assert correction is not None
        assert hashlib.sha256((analysis/'raw_run_runner_source.txt').read_bytes()).hexdigest()==manifest['runner_sha256']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==correction['corrected_runner_sha256']
    from rl import PolicyNet
    from experiments.developmental_care import trajectory,policy_hash
    replays=0;torch.set_num_threads(1)
    for s,obj in itertools.product(range(10),('survival','offspring')):
        checkpoint=ROOT/f'results/final_reward_v1/{obj}_s{s}.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==manifest['checkpoint_sha256'][checkpoint.name]
        policy=PolicyNet(seed=s);policy.load_state_dict(torch.load(checkpoint,weights_only=True));policy.eval()
        for p in policy.parameters():p.requires_grad_(False);assert torch.isfinite(p).all()
        original=json.loads((ROOT/f'results/final_reward_v1/{obj}_s{s}_evaluation.json').read_text())['trained_policy_hash']
        assert policy_hash(policy)==original
        if replay:
            for c,d in ((False,False),(True,True)):
                actual=json.loads(json.dumps(trajectory(policy,19,c,d,full=True)))
                expected_row=records[(s,obj,19,c,d)]
                assert all(actual[k]==v for k,v in expected_row.items() if k not in ('training_seed','objective'))
                assert policy_hash(policy)==original;replays+=1
    return {'trajectories':1600,'files':1602,'controls_identical_contexts':400,
            'sensory_audits':96000,'verified_checkpoints':20,'matched_replays':replays,
            'statistical_contrasts':checked,'all_integrity_checks_passed':True,
            'metadata_correction_required':correction is not None}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'results/developmental_care_v1')
    parser.add_argument('--analysis',type=Path,default=ROOT/'results/developmental_care_v1_analysis')
    parser.add_argument('--replay',action='store_true');parser.add_argument('--out',type=Path)
    args=parser.parse_args();result=verify(args.data,args.analysis,args.replay)
    if args.out:
        with args.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
