"""Regenerate verified Phase 1/2 report figures into a NEW directory only."""
import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='alife-fig-'))
os.environ.setdefault('XDG_CACHE_HOME',tempfile.mkdtemp(prefix='alife-cache-'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from verify_completed import ROOT,load,verify

COLORS={'survival':'#4477AA','offspring':'#CC6677'}
LABELS={'survival':'Survival only','offspring':'Survival + offspring'}


def generate(out):
    verification=verify()
    out.mkdir(parents=True,exist_ok=False)
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    phase1=load(ROOT/'results/final_reward_v1/summary.json')
    phase2=load(ROOT/'results/controlled_care_v1_analysis/analysis.json')
    indices=np.random.default_rng(20261014).integers(0,10,size=(100000,10))
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,name):
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(out/f'{name}.png',dpi=300,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.6,2.9))
    pairs=phase1['paired_primary_endpoint']
    for p in pairs:ax.plot([0,1],[100*p['survival'],100*p['offspring']],color='#AAAAAA',lw=.8,zorder=1)
    for x,obj in enumerate(('survival','offspring')):
        y=np.array([p[obj] for p in pairs])*100
        ax.scatter(np.full(10,x),y,color=COLORS[obj],s=24,zorder=2)
        ax.scatter([x],[y.mean()],marker='D',color='black',s=25,zorder=3)
    ax.set_xticks([0,1],['Survival\nonly','Survival +\noffspring'])
    ax.set_ylabel('Offspring independence (%)');ax.set_ylim(0,100)
    ax.set_title('Phase 1: ten paired training seeds')
    save(fig,'phase1_paired_independence')
    fig,axes=plt.subplots(1,2,figsize=(7,2.9),sharey=True)
    for ax,energy in zip(axes,(10,20)):
        for obj in ('survival','offspring'):
            groups=[next(g for g in phase2['groups'] if g['objective']==obj and g['energy']==energy and g['background']==n and g['disruption']==d)
                    for n,d in ((0,False),(0,True),(8,False),(8,True))]
            means=[];errors=[]
            for g in groups:
                values=np.array(g['seed_rates']);ci=np.quantile(values[indices].mean(axis=1),[.025,.975])
                means.append(values.mean()*100);errors.append([(values.mean()-ci[0])*100,(ci[1]-values.mean())*100])
            ax.errorbar(range(4),means,yerr=np.array(errors).T,fmt='o-',capsize=3,
                        color=COLORS[obj],label=LABELS[obj])
        ax.set_xticks(range(4),['0 / OFF','0 / ON','8 / OFF','8 / ON'],rotation=25,ha='right')
        ax.set_title(f'Parent energy {energy}');ax.set_ylim(-2,102)
        ax.set_xlabel('Background bodies / care disruption')
    axes[0].set_ylabel('Offspring independence (%)');axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout();save(fig,'phase2_independence')
    fig,axes=plt.subplots(1,2,figsize=(7,2.9),sharey=True)
    for ax,obj in zip(axes,('survival','offspring')):
        groups=[next(g for g in phase2['groups'] if g['energy'] is None and g['objective']==obj and g['background']==n and g['disruption']==d)
                for n,d in ((0,False),(0,True),(8,False),(8,True))]
        attended=np.array([g['counts']['attended_ticks']/g['counts']['care_ticks'] for g in groups])*100
        blocked=np.array([g['counts']['crowd_blocked_ticks']/g['counts']['care_ticks'] for g in groups])*100
        ax.bar(range(4),attended,color='#228833',label='Attended')
        ax.bar(range(4),blocked,bottom=attended,color='#CC3311',label='Crowd-blocked')
        ax.bar(range(4),100-attended-blocked,bottom=attended+blocked,color='#BBBBBB',label='Parent unavailable / out of range')
        ax.set_xticks(range(4),['0 / OFF','0 / ON','8 / OFF','8 / ON'],rotation=25,ha='right')
        ax.set_title(LABELS[obj]);ax.set_ylim(0,100);ax.set_xlabel('Background bodies / care disruption')
    axes[0].set_ylabel('Observed dependent-pup ticks (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,
               frameon=False,fontsize=8,bbox_to_anchor=(.5,1.06))
    fig.tight_layout();save(fig,'phase2_care_mechanism')
    fig,ax=plt.subplots(figsize=(3.6,2.9))
    effects={obj:next(t['seed_effects'] for t in phase2['paired_contrasts'] if t['energy'] is None and t['contrast']==f'disruption_effect_{obj}_background_8')
             for obj in ('survival','offspring')}
    for s,o in zip(effects['survival'],effects['offspring']):ax.plot([0,1],[s*100,o*100],color='#AAAAAA',lw=.8)
    for x,obj in enumerate(('survival','offspring')):ax.scatter(np.full(10,x),np.array(effects[obj])*100,color=COLORS[obj])
    ax.axhline(0,color='black',lw=.7);ax.set_xticks([0,1],['Survival\nonly','Survival +\noffspring'])
    ax.set_ylabel('Independence change, ON − OFF (pp)');ax.set_ylim(-85,3)
    ax.set_title('High crowding; survival-policy floor')
    fig.tight_layout();save(fig,'supplement_paired_penalties')
    captions={
        'phase1_paired_independence':'Each line joins one independent training-seed pair. Each point averages 40 assigned-pup trials (20 fixed worlds × two energies); diamonds are ten-seed means. These are actual independence outcomes, not directional pup seeking. Parent follow-up ends at pup death or tick 60.',
        'phase2_independence':'Each cell has 200 trials (ten training seeds × 20 fixed worlds). Error bars: unadjusted 95% percentile bootstrap intervals over ten training seeds, conditional on the world panel. Background bodies are stationary, resource-neutral evaluation proxies; OFF/ON refers to the programmed care-disruption flag. All energy-20 parents survive.',
        'phase2_care_mechanism':'Both energies pooled; 400 trials per bar. Denominator is observed dependent-pup care ticks, ending at death or maturation (maximum 59 per trial). Red is baseline attendance invalidated by active crowding disruption. Grey includes out-of-range or dead parents. This denominator varies by survival and is not a common 60-tick mortality risk.',
        'supplement_paired_penalties':'Lines join the two trained policies within each seed. Units are percentage points, not relative percentages. Survival policies reach zero independence in high/ON; their lower baseline bounds the absolute penalty. This plot cannot establish intrinsically greater offspring-policy susceptibility.'}
    (out/'CAPTIONS.json').write_text(json.dumps(captions,indent=2))
    (out/'CAPTIONS.md').write_text('\n\n'.join(f'**{name}**\n\n{text}' for name,text in captions.items())+'\n')
    print(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    generate(Path(parser.parse_args().out))
