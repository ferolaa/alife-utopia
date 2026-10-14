"""Prespecified paired-seed analysis and figures for frozen neural evaluation."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='neural-fig-'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELLS = ['initial','survival_normal','offspring_normal','survival_masked','offspring_masked']
COMPARISONS = {'survival_training': ('survival_normal','initial'),
               'offspring_training': ('offspring_normal','initial'),
               'survival_masking': ('survival_masked','survival_normal'),
               'offspring_masking': ('offspring_masked','offspring_normal')}


def analyze(data, out):
    data,out=Path(data),Path(out)
    if out.exists(): raise FileExistsError('Analysis target must be new')
    manifest=json.loads((data/'manifest.json').read_text())
    records=json.loads((data/'records.json').read_text())
    assert manifest['training_seeds']==list(range(10)) and manifest['world_seeds']==list(range(100000,100020))
    expected=set(itertools.product(range(10),range(100000,100020),(10,20),CELLS))
    assert len(records)==len(expected)==2000
    assert {(r['training_seed'],r['world_seed'],r['energy'],r['cell']) for r in records}==expected
    rates={c:np.array([sum(r['outcome']['independent'] for r in records if r['cell']==c and r['training_seed']==s)/40 for s in range(10)]) for c in CELLS}
    indices=np.random.default_rng(20261014).integers(0,10,(100000,10))
    signs=np.array(list(itertools.product((-1,1),repeat=10)))
    contrasts=[]
    for name,(a,b) in COMPARISONS.items():
        x=rates[a]-rates[b]
        contrasts.append({'contrast':name,'seed_effects':x.tolist(),'mean':float(x.mean()),
            'ci95':np.quantile(x[indices].mean(1),[.025,.975]).tolist(),
            'exact_p':float(np.mean(abs(signs@x/10)>=abs(x.mean())-1e-12)),
            'negative_seeds':int(sum(x < -1e-12)), 'zero_seeds':int(sum(abs(x)<=1e-12)),
            'positive_seeds':int(sum(x > 1e-12))})
    running=0
    for i,t in enumerate(sorted(contrasts,key=lambda t:t['exact_p'])):
        running=max(running,(4-i)*t['exact_p']);t['holm_p']=min(1.,running)
    groups=json.loads((data/'summary.json').read_text())['groups']
    seed_rows=[{'seed':s,**{c:float(rates[c][s]) for c in CELLS},**{t['contrast']:t['seed_effects'][s] for t in contrasts}} for s in range(10)]
    result={'contrasts':contrasts,'seed_results':seed_rows,'groups':groups,
        'method':'ten paired training seeds; 100000 paired percentile bootstrap resamples, RNG 20261014; 1024 exact two-sided sign flips; Holm across four prespecified contrasts; fixed-world conditional unadjusted 95% intervals',
        'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()},
        'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.mkdir(parents=True)
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'training_seed_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(seed_rows[0]),lineterminator='\n');w.writeheader();w.writerows(seed_rows)
    lines=['# Frozen neural-policy evaluation','',
        '| Policy | Energy | Independence | Attendance | Meals | Neglect deaths | Parent deaths | Alive early stop | Full survival | Posthumous success |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['cell']} | {g['energy'] or 'pooled'} | {g['successes']}/{g['trials']} ({g['success_rate']*100:.2f}%) | {g['attendance_fraction']*100:.2f}% | {g['parent_meals']} | {g['pup_deaths_by_cause']['neglect']} | {g['parent_deaths']} | {g['parent_alive_early_stop']} | {g['parent_survived_full_trial']} | {g['posthumous_independence']} |")
    lines+=['','| Contrast | Difference pp | 95% interval pp | Exact p | Holm p | Negative / zero / positive seeds |','|---|---:|---|---:|---:|---|']
    for t in contrasts:
        lines.append(f"| {t['contrast']} | {100*t['mean']:+.2f} | {100*t['ci95'][0]:+.2f}, {100*t['ci95'][1]:+.2f} | {t['exact_p']:.6f} | {t['holm_p']:.6f} | {t['negative_seeds']} / {t['zero_seeds']} / {t['positive_seeds']} |")
    lines+=['','Two thousand unique trials; ten unique initial networks and twenty trained networks. Shared normal trained and initial controls are not counted as independent copies.','',result['method'], '',
        'Care attendance is pooled over observed care ticks; meals and parent survival have unequal follow-up because the original protocol stops at pup death. Parent deaths and posthumous successes remain explicit. This measures gradient-training-associated caregiving performance and functional sensitivity to removed own-pup information, not hidden representations. Masking induces an input-distribution shift. No training occurred during evaluation.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7,2.8))
    colors={'initial':'#666666','survival_normal':'#4477AA','offspring_normal':'#CC6677'}
    for i,c in enumerate(CELLS[:3]):
        x=rates[c]*100;ci=np.quantile(x[indices].mean(1),[.025,.975]);mean=x.mean()
        axes[0].scatter(np.full(10,i)+np.linspace(-.10,.10,10),x,color=colors[c],alpha=.45,s=14)
        axes[0].errorbar(i,mean,yerr=[[mean-ci[0]],[ci[1]-mean]],fmt='D',color=colors[c],capsize=4)
    axes[0].set_xticks(range(3),['Untrained','Survival\ntrained','Offspring\ntrained'])
    for obj,offset in [('survival',-.06),('offspring',.06)]:
        col=colors[obj+'_normal']
        for s in range(10):axes[1].plot(np.array([0,1])+offset,[rates[obj+'_normal'][s]*100,rates[obj+'_masked'][s]*100],color=col,alpha=.22,lw=.8)
        means=[];errors=[]
        for suffix in ['normal','masked']:
            x=rates[obj+'_'+suffix]*100;ci=np.quantile(x[indices].mean(1),[.025,.975]);means.append(x.mean());errors.append([x.mean()-ci[0],ci[1]-x.mean()])
        axes[1].errorbar(np.array([0,1])+offset,means,yerr=np.array(errors).T,fmt='D-',color=col,capsize=4,label=obj.capitalize()+' trained')
    axes[1].set_xticks([0,1],['Normal inputs','Own-pup inputs\nmasked'])
    axes[1].legend(frameon=False,fontsize=7)
    for ax in axes:ax.set_ylim(0,100);ax.set_ylabel('Assigned-pup independence (%)')
    axes[0].set_title('A  Gradient training versus initialization');axes[1].set_title('B  Frozen-policy sensory ablation')
    fig.tight_layout();fig.savefig(out/'neural_policy.pdf',bbox_inches='tight');fig.savefig(out/'neural_policy.png',dpi=300,bbox_inches='tight');plt.close(fig)
    (out/'CAPTION.md').write_text('Frozen policies, original matched assigned-pup assay; energies10/20 equally weighted. Each condition contains400 trials, with ten training seeds as inferential units. Small points/lines show individual training seeds; diamonds and bars show means and unadjusted95% paired-seed percentile bootstrap intervals conditional on twenty held-out worlds. Initial policies are shared, not duplicate replicates. Masking zeros only own-pup direction and need inputs. Independence can occur after parent death; success does not mean continuous parental attendance. Input masking creates a distribution shift and does not identify hidden representations.\n')
    print(json.dumps(contrasts,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'results/neural_policy_v1');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.data,a.out)
