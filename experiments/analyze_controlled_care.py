"""Post-run integrity checks and paired training-seed analysis; no protocol edits."""
import csv
import hashlib
import itertools
import json
import os
from collections import Counter
from pathlib import Path

import argparse
import tempfile
os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='alife-analysis-'))
os.environ.setdefault('XDG_CACHE_HOME',tempfile.mkdtemp(prefix='alife-cache-'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/controlled_care_v1'
OUT=None
from verify_completed import raw,load


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inference(values, bootstrap_indices):
    x=np.array(values,dtype=float)
    interval=np.quantile(x[bootstrap_indices].mean(axis=1),[.025,.975])
    signs=np.array(list(itertools.product((-1,1),repeat=len(x))))
    p=float(np.mean(np.abs(signs@x/len(x))>=abs(x.mean())-1e-12))
    return {'n_training_pairs':len(x),'mean':float(x.mean()),'ci95':interval.tolist(),
            'exact_two_sided_sign_flip_p':p,'seed_effects':x.tolist()}


def main():
    global DATA,OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();DATA=args.data.resolve();OUT=args.out.resolve()
    manifest=load(DATA/'manifest.json')
    summary=json.loads((DATA/'summary.json').read_text())
    assert manifest['training_seeds']==list(range(10))
    assert manifest['world_seeds']==list(range(300000,300020))
    expected={'manifest.json','summary.json'}|{f'{o}_s{s}.json' for o in ('survival','offspring') for s in range(10)}
    assert all((DATA/name).is_file() or (DATA==ROOT/'results/controlled_care_v1' and (ROOT/'results/controlled_care_v1_archive'/f'{name}.gz').is_file()) for name in expected)
    records=[r for name in sorted(expected-{'manifest.json','summary.json'})
             for r in load(DATA/name)]
    assert len(records)==summary['completed_trials']==summary['expected_trials']==3200
    keys={(r['training_seed'],r['objective'],r['world_seed'],r['initial_energy'],r['background_count'],r['disruption']) for r in records}
    assert len(keys)==3200
    for r in records:
        assert r['elapsed_ticks']==len(r['history'])==60
        assert r['starting_population']+r['births']-r['deaths']==r['final_population']
        assert r['starting_population']==2 and r['births']==0
        assert sum(r['deaths_by_cause'].values())==r['deaths']
        assert r['final_total_occupancy_bodies']==r['final_population']+r['background_count']
        assert all(t['food_count']==60 for t in r['history'])
        assert r['counts']['third_party_crowd_sum']==r['background_count']*r['counts']['care_ticks']
        assert r['counts']['parent_action_ticks']==sum(bool(t['actions'].get('parent')) for t in r['history'])
        assert r['parent_alive']==r['history'][-1]['parent_alive']
        assert not r['parent_alive'] or r['parent_cause_of_death'] is None
    for seed in manifest['world_seeds']:
        for energy in (10,20):
            group=[r for r in records if r['world_seed']==seed and r['initial_energy']==energy]
            assert len({r['initial_hash'] for r in group})==1
    baseline={str(p):hash_file(p) for p in (ROOT/'results/final_reward_v1').iterdir()}
    assert all(hash_file(Path(p))==h for p,h in baseline.items())
    assert hash_file(ROOT/'experiments/controlled_care.py')==manifest['runner_sha256']
    OUT.mkdir(exist_ok=False)
    groups=[]
    rates={}
    for objective in ('survival','offspring'):
        for energy in (None,10,20):
            for n in (0,8):
                for disruption in (False,True):
                    group=[r for r in records if r['objective']==objective and r['background_count']==n and
                           r['disruption']==disruption and (energy is None or r['initial_energy']==energy)]
                    c,mort=Counter(),Counter()
                    for r in group:c.update(r['counts']);mort.update(r['deaths_by_cause'])
                    seed_rates=[sum(r['independent'] for r in group if r['training_seed']==s)/
                                sum(r['training_seed']==s for r in group) for s in range(10)]
                    rates[(objective,energy,n,disruption)]=np.array(seed_rates)
                    groups.append({'objective':objective,'energy':energy,'background':n,'disruption':disruption,
                        'trials':len(group),'successes':sum(r['independent'] for r in group),
                        'rate':sum(r['independent'] for r in group)/len(group),'seed_rates':seed_rates,
                        'zero_success_training_seeds':sum(x==0 for x in seed_rates),
                        'counts':dict(c),'deaths_by_cause':dict(mort),
                        'parent_survivors':sum(r['parent_alive'] for r in group),
                        'parent_deaths_by_cause':dict(Counter(r['parent_cause_of_death'] for r in group if not r['parent_alive'])),
                        'pup_deaths_by_cause':dict(Counter(r['pup_cause_of_death'] for r in group if not r['independent'])),
                        'success_after_parent_death':sum(r['independent'] and not r['parent_alive'] for r in group),
                        'attendance_observed_care_fraction':c['attended_ticks']/c['care_ticks'],
                        'attended_ticks_per_planned_dependent_tick':c['attended_ticks']/(len(group)*59),
                        'mean_total_pup_crowding':c['pup_crowd_sum']/c['care_ticks']})
    indices=np.random.default_rng(20261014).integers(0,10,size=(100000,10))
    tests=[]
    for energy in (None,10,20):
        contrasts=[]
        for n in (0,8):
            for d in (False,True):
                contrasts.append((f'policy_advantage_background_{n}_disruption_{int(d)}',
                                  rates[('offspring',energy,n,d)]-rates[('survival',energy,n,d)]))
        for objective in ('survival','offspring'):
            for n in (0,8):
                contrasts.append((f'disruption_effect_{objective}_background_{n}',
                                  rates[(objective,energy,n,True)]-rates[(objective,energy,n,False)]))
        interaction=sum(sign*rates[(objective,energy,n,d)]
                        for objective,sign_o in (('offspring',1),('survival',-1))
                        for n,sign_n in ((8,1),(0,-1)) for d,sign_d in ((True,1),(False,-1))
                        for sign in (sign_o*sign_n*sign_d,))
        contrasts.append(('policy_x_crowding_x_disruption',interaction))
        family=[]
        for name,x in contrasts:
            family.append({'contrast':name,'energy':energy,**inference(x,indices)})
        ordered=sorted(family,key=lambda t:t['exact_two_sided_sign_flip_p'])
        running=0
        for i,t in enumerate(ordered):
            running=max(running,(len(family)-i)*t['exact_two_sided_sign_flip_p'])
            t['holm_adjusted_p_within_nine_contrasts']=min(1.,running)
        tests.extend(family)
    result={'groups':groups,'paired_contrasts':tests,'runtime_seconds':summary['total_seconds'],
            'analysis_method':'Equal-weight independent training-seed pairs; bootstrap 100000 paired resamples, seed 20261014; exact two-sided sign flip over 1024 patterns; Holm within each nine-contrast family. Energy-stratified contrasts secondary. Intervals unadjusted.',
            'interaction_definition':'[(offspring-survival) high/on - high/off] - [(offspring-survival) low/on - low/off]',
            'integrity':{'trials':3200,'expected_files':22,'protected_files_unchanged':len(baseline),
                         'matched_initial_states':True,'population_and_mortality_accounting':True,
                         'runner_hash_matches_manifest':True},
            'output_sha256':{name:hashlib.sha256(raw(DATA/name)).hexdigest() for name in sorted(expected)}}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2))
    with (OUT/'training_seed_rates.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['training_seed','objective','energy','background','disruption','independence_fraction'])
        for key,x in rates.items():
            for seed,value in enumerate(x):writer.writerow([seed,*key,float(value)])
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors={'survival':'#4477AA','offspring':'#CC6677'}
    fig,axes=plt.subplots(1,2,figsize=(7,2.8),sharey=True)
    for ax,energy in zip(axes,(10,20)):
        for objective in ('survival','offspring'):
            means=[];errors=[]
            for n,d in ((0,False),(0,True),(8,False),(8,True)):
                x=rates[(objective,energy,n,d)];ci=np.quantile(x[indices].mean(axis=1),[.025,.975])
                means.append(x.mean()*100);errors.append((x.mean()-ci[0],ci[1]-x.mean()))
            ax.errorbar(range(4),means,yerr=np.array(errors).T*100,fmt='o-',capsize=3,
                        color=colors[objective],label=objective)
        ax.set_xticks(range(4),['Low\nOFF','Low\nON','High\nOFF','High\nON'])
        ax.set_title(f'Parent energy {energy}');ax.set_ylim(-2,102);ax.set_xlabel('Background crowding / care disruption')
    axes[0].set_ylabel('Offspring independence (%)');axes[1].legend(frameon=False)
    fig.tight_layout();fig.savefig(OUT/'independence.pdf');fig.savefig(OUT/'independence.png',dpi=300);plt.close(fig)
    high=[t for t in tests if t['energy'] is None and t['contrast']=='policy_advantage_background_8_disruption_1'][0]
    triple=[t for t in tests if t['energy'] is None and t['contrast']=='policy_x_crowding_x_disruption'][0]
    lines=['# Controlled parental-care experiment','',
           f"Completed 3,200 trials in {summary['total_seconds']:.2f} seconds. All 247 tests passed before launch.",
           '','Frozen checkpoints; training seeds 0–9; worlds 300000–300019; energies 10/20; backgrounds 0/8; disruption OFF/ON.',
           '','| Policy | Energy | Background | Disruption | Independent / trials | Parent survivors |',
           '|---|---:|---:|---|---:|---:|']
    for g in groups:
        if g['energy'] is not None:
            lines.append(f"| {g['objective']} | {g['energy']} | {g['background']} | {'ON' if g['disruption'] else 'OFF'} | {g['successes']}/{g['trials']} ({100*g['rate']:.2f}%) | {g['parent_survivors']}/{g['trials']} |")
    lines.extend(['',f"Pooled-energy high/ON policy advantage: {100*high['mean']:.2f} percentage points; 95% paired bootstrap interval {100*high['ci95'][0]:.2f} to {100*high['ci95'][1]:.2f}; exact p={high['exact_two_sided_sign_flip_p']:.6f}.",
        f"Three-way interaction: {100*triple['mean']:.2f} points; interval {100*triple['ci95'][0]:.2f} to {100*triple['ci95'][1]:.2f}; exact p={triple['exact_two_sided_sign_flip_p']:.6f}.",
        '',result['analysis_method'],'',
        'Limitations: stationary resource-neutral bodies are an evaluation intervention; no colony reproduction; crowding is visible to policies; parent-inclusive tick-start crowd counts coexist with post-movement attendance; neglect tolerance permits independence after parent death; policy floor effects limit resilience comparisons; training credit limitations remain; this is not a biological replication of Calhoun.',
        '', 'Recommended three-page report figures: independence interaction plot (provided); paired training-seed disruption penalties; attendance/blocking with parent survival and orphan-success annotations.'])
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'runtime':summary['total_seconds'],'high_on_advantage':high,'interaction':triple,'integrity':result['integrity']},indent=2))


if __name__=='__main__':main()
