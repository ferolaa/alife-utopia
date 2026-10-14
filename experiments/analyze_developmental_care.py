"""Read-only Phase 3 analysis; paired training seeds, fixed world panel, new outputs."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='alife-phase3-'))
os.environ.setdefault('XDG_CACHE_HOME',tempfile.mkdtemp(prefix='alife-cache-'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inference(values,indices):
    x=np.asarray(values,dtype=float)
    ci=np.quantile(x[indices].mean(axis=1),[.025,.975])
    permutations=np.array(list(itertools.product((-1,1),repeat=10)))@x/10
    p=float(np.mean(np.abs(permutations)>=abs(x.mean())-1e-12))
    return {'mean':float(x.mean()),'ci95':ci.tolist(),'exact_two_sided_sign_flip_p':p,
            'seed_effects':x.tolist(),'negative_seeds':int(sum(x<0)),
            'zero_seeds':int(sum(np.isclose(x,0,atol=1e-12))),
            'positive_seeds':int(sum(x>0))}


def analyze(data,out):
    data,out=Path(data).resolve(),Path(out).resolve()
    manifest=json.loads((data/'manifest.json').read_text())
    summary=json.loads((data/'summary.json').read_text())
    assert manifest['training_seeds']==list(range(10))
    assert manifest['adult_world_seeds']==list(range(400000,400020))
    assert manifest['childhood_world_seeds']==list(range(410000,410020))
    assert manifest['bridge_world_seeds']==list(range(420000,420020))
    paths=sorted(data.glob('*_s*.json'))
    records=[json.loads(p.read_text()) for p in paths]
    assert len(records)==summary['trajectories']==1600
    assert len(list(data.iterdir()))==1602
    assert all(summary['technical_gates'].values())
    groups=[];rates={}
    for obj in ('survival','offspring'):
        for childhood in (False,True):
            for damage in (False,True):
                group=[r for r in records if r['objective']==obj and r['disrupted_childhood']==childhood and r['damage_enabled']==damage]
                assert len(group)==200 and all(r['adult'] is not None for r in group)
                counts,mortality=Counter(),Counter()
                for r in group:counts.update(r['adult']['counts']);mortality.update(r['adult']['deaths_by_cause'])
                values=[sum(r['adult']['independent'] for r in group if r['training_seed']==s)/20 for s in range(10)]
                rates[(obj,childhood,damage)]=np.array(values)
                g={'objective':obj,'disrupted_childhood':childhood,'damage_enabled':damage,
                   'assigned_cohort':200,'childhood_survivors':sum(r['childhood_survivors'] for r in group),
                   'mature_survivors':sum(r['mature_survivors'] for r in group),'adults_evaluated':200,
                   'independence_successes':sum(r['adult']['independent'] for r in group),
                   'independence_fraction':float(np.mean(values)),'seed_rates':values,
                   'parent_survivors':sum(r['adult']['parent_alive'] for r in group),
                   'counts':dict(counts),'deaths_by_cause':dict(mortality),
                   'attendance_fraction':counts['attended_ticks']/counts['care_ticks'],
                   'attended_per_planned_care_tick':counts['attended_ticks']/(200*59),
                   'impairments':sorted({r['adult']['impairment'] for r in group}),
                   'independence_after_parent_death':sum(r['adult']['success_after_parent_death'] for r in group)}
                groups.append(g)
    effects={obj:(rates[(obj,True,True)]-rates[(obj,False,True)])-
             (rates[(obj,True,False)]-rates[(obj,False,False)]) for obj in ('survival','offspring')}
    effects['objective_interaction']=effects['offspring']-effects['survival']
    indices=np.random.default_rng(20261014).integers(0,10,size=(100000,10))
    tests=[{'contrast':name,**inference(x,indices)} for name,x in effects.items()]
    running=0
    for i,t in enumerate(sorted(tests,key=lambda t:t['exact_two_sided_sign_flip_p'])):
        running=max(running,(3-i)*t['exact_two_sided_sign_flip_p'])
        t['holm_adjusted_p']=min(1.,running)
    seed_rows=[]
    for s in range(10):
        row={'training_seed':s}
        for obj in ('survival','offspring'):
            row[f'{obj}_control']=float(rates[(obj,False,False)][s])
            row[f'{obj}_impaired']=float(rates[(obj,True,True)][s])
            row[f'{obj}_effect']=float(effects[obj][s])
        row['objective_interaction']=float(effects['objective_interaction'][s]);seed_rows.append(row)
    out.mkdir(parents=True,exist_ok=False)
    flips={}
    for obj in ('survival','offspring'):
        lost=gained=0
        for s in range(10):
            for j in range(20):
                group=[r for r in records if r['objective']==obj and r['training_seed']==s and r['world_index']==j]
                base=next(r for r in group if not r['disrupted_childhood'] and not r['damage_enabled'])['adult']['independent']
                impaired=next(r for r in group if r['disrupted_childhood'] and r['damage_enabled'])['adult']['independent']
                lost+=base and not impaired;gained+=impaired and not base
        flips[obj]={'lost_successes':int(lost),'gained_successes':int(gained)}
    result={'groups':groups,'paired_contrasts':tests,'seed_results':seed_rows,'matched_flips':flips,
            'total_seconds':summary['total_seconds'],'statistical_unit':'ten independent training-seed pairs',
            'effect_definition':'(disrupted/ON - normal/ON) - (disrupted/OFF - normal/OFF)',
            'interaction_definition':'offspring developmental effect minus survival developmental effect',
            'analysis_method':'100000 paired seed bootstrap resamples; RNG 20261014; unadjusted percentile 95% intervals conditional on fixed twenty-world panel; all 1024 exact two-sided sign flips; Holm across only three planned contrasts',
            'technical_gates':summary['technical_gates'],
            'source_sha256':sha(Path(__file__)),
            'input_sha256':{p.name:sha(p) for p in data.iterdir()}}
    (out/'analysis.json').write_text(json.dumps(result,indent=2))
    with (out/'training_seed_effects.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(seed_rows[0]),lineterminator='\n');w.writeheader();w.writerows(seed_rows)
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors={'survival':'#4477AA','offspring':'#CC6677','objective_interaction':'#228833'}
    labels={'survival':'Survival only','offspring':'Survival + offspring','objective_interaction':'Between objectives'}
    fig,axes=plt.subplots(1,2,figsize=(7,3.1),gridspec_kw={'width_ratios':[1.15,1]})
    for obj in ('survival','offspring'):
        means=[];errors=[]
        for c,d in ((False,False),(False,True),(True,False),(True,True)):
            x=rates[(obj,c,d)];ci=np.quantile(x[indices].mean(axis=1),[.025,.975])
            means.append(x.mean()*100);errors.append([(x.mean()-ci[0])*100,(ci[1]-x.mean())*100])
        axes[0].errorbar(range(4),means,yerr=np.array(errors).T,fmt='o-',capsize=3,color=colors[obj],label=labels[obj])
    axes[0].set_xticks(range(4),['Normal\nOFF','Normal\nON','Disrupted\nOFF','Disrupted\nON'])
    axes[0].set_ylabel('Assigned-pup independence (%)');axes[0].set_ylim(0,100)
    axes[0].set_xlabel('Childhood care / impairment assignment');axes[0].legend(frameon=False,fontsize=7)
    for i,t in enumerate(tests):
        x=np.array(t['seed_effects'])*100
        axes[1].scatter(x,np.full(10,i)+np.linspace(-.09,.09,10),s=12,alpha=.5,color=colors[t['contrast']])
        axes[1].errorbar(t['mean']*100,i,xerr=np.array([[t['mean']-t['ci95'][0]],[t['ci95'][1]-t['mean']]])*100,
                         fmt='D',color=colors[t['contrast']],capsize=3)
    axes[1].set_yticks(range(3),[labels[t['contrast']] for t in tests]);axes[1].invert_yaxis()
    axes[1].axvline(0,color='black',lw=.7);axes[1].set_xlabel('Developmental contrast (pp)')
    axes[0].set_title('A  Uncrowded adult assessment');axes[1].set_title('B  Ten paired training seeds')
    fig.tight_layout();fig.savefig(out/'phase3.pdf',bbox_inches='tight');fig.savefig(out/'phase3.png',dpi=300,bbox_inches='tight');plt.close(fig)
    caption='Frozen policies; childhood exposure fixed at 18 unattended ticks. Adult assessment: age 120, energy 20, no background bodies or active care disruption. Each cell has 200 evaluations but inference uses ten training-seed pairs. Three zero-impairment controls are identical. Diamonds show paired means, small points individual seed effects; error bars are unadjusted 95% paired bootstrap intervals conditional on the fixed world panel. Negative contrasts indicate fewer independent assigned pups. Persistence and attenuation are programmed, not learned neural damage.'
    (out/'CAPTION.md').write_text(caption+'\n')
    report=['# Full Phase 3: programmed neglect-dependent cue attenuation','',
            f"1,600 complete trajectories; runtime {summary['total_seconds']:.2f} seconds. Preflight: 265 tests passed.",'',
            '| Objective | Childhood | Assignment | Successful / assessed | Attendance | Meals | Neglect deaths |',
            '|---|---|---|---:|---:|---:|---:|']
    for g in groups:
        report.append(f"| {g['objective']} | {'disrupted' if g['disrupted_childhood'] else 'normal'} | {'ON' if g['damage_enabled'] else 'OFF'} | {g['independence_successes']}/200 ({100*g['independence_fraction']:.2f}%) | {100*g['attendance_fraction']:.2f}% | {g['counts']['parent_meals']} | {g['deaths_by_cause']['neglect']} |")
    report+=['','| Planned contrast | Effect (pp) | 95% paired interval | Exact p | Holm p |','|---|---:|---|---:|---:|']
    for t in tests:
        report.append(f"| {t['contrast']} | {100*t['mean']:.2f} | {100*t['ci95'][0]:.2f}, {100*t['ci95'][1]:.2f} | {t['exact_two_sided_sign_flip_p']:.6f} | {t['holm_adjusted_p']:.6f} |")
    report+=['','## Individual training-seed results','','| Seed | Survival control | Survival impaired | Survival effect pp | Offspring control | Offspring impaired | Offspring effect pp | Interaction pp |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in seed_rows:
        report.append('| '+str(r['training_seed'])+' | '+' | '.join(f"{100*r[k]:.1f}" for k in list(r)[1:])+' |')
    report+=['',caption,'',result['analysis_method'], '',
        'Interpretation must distinguish programmed persistence from its measured downstream effect on policy performance. Missing adult assessments cannot be silently classified as adult care failures. Observed-care attendance ends at pup death/maturation; mortality uses full parental follow-up. Modelled cue scaling is not trauma, biological recovery, inherited learning or changed network weights. The first three worlds and first two training pairs were used in the pilot; no parameters changed afterward.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'runtime':summary['total_seconds'],'tests':tests,'matched_flips':flips},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'results/developmental_care_v1')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();analyze(args.data,args.out)
