"""The fixed eight-run pilot, with no search, PPO, or alternative architectures."""
from dataclasses import asdict, replace
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from config import UNIVERSE_25
from .care import assigned_pup_trial
from .data import RewardConfig
from .evaluation import SeedBlocks, state_hash
from .experiment import Experiment, provenance
from .learning import LearnerConfig

TARGET = 200_000
INTERVAL = 50_000
MAX_OVERSHOOT = 20_000
MAX_SPREAD = 20_000
VALIDATION_SEEDS = tuple(range(3_000_100, 3_000_105))
ENERGIES = (10, 20)
CELLS = tuple(itertools.product(('reinforce', 'reinforce_value'), ('survival', 'offspring'), (0, 1)))
BASELINE = 'ce129bef00de069d171c99b797f142f0711fb626'
PHASE1 = '66fe988f5774c744f245db2270278fb69747896a'


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def historical_integrity(root):
    names = subprocess.check_output(['git', '-C', str(root), 'ls-tree', '-r', '--name-only', BASELINE], text=True).splitlines()
    receipts = {}
    for name in names:
        expected = subprocess.check_output(['git', '-C', str(root), 'cat-file', 'blob', f'{BASELINE}:{name}'])
        current = (root/name).read_bytes()
        if current != expected:
            raise RuntimeError(f'Historical file changed: {name}')
        receipts[name] = hashlib.sha256(current).hexdigest()
    if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip() != PHASE1:
        raise RuntimeError('pilot must run on the reviewed Phase 1 commit with uncommitted corrections')
    if subprocess.check_output(['git','-C',str(root),'branch','--show-current'],text=True).strip() != 'dlai-revision-v2':
        raise RuntimeError('wrong branch')
    return receipts


def training_config():
    # Exact historical completed training preset, not the long continuous-colony run.
    return UNIVERSE_25.variant('v2-controlled-pilot',n_ticks=500,n_initial_agents=40,max_population=120)


def protocol():
    cfg = training_config()
    if cfg.crowding_blocks_care or cfg.nest_intrusion_enabled or cfg.developmental_damage_enabled:
        raise RuntimeError('pilot cannot introduce care disruption or impairment')
    return dict(schema=1, purpose='controlled learning pilot only; descriptive n=2',
        cells=[dict(algorithm=a,objective=o,seed=s) for a,o,s in CELLS],
        environment=asdict(cfg),architecture='original',actor_parameters=234,critic_parameters=5057,
        reward_scope='colony',gamma=.99,survival_reward=.01,independence_reward=1.,
        actor_lr=.001,value_lr=.001,entropy_coefficient=0.,max_grad_norm=None,
        target_training_decisions=TARGET,validation_interval_decisions=INTERVAL,
        validation_rule='initial; first completed-episode boundary crossing each threshold; actual counts recorded',
        max_overshoot_decisions=MAX_OVERSHOOT,max_run_budget_spread=MAX_SPREAD,
        overshoot_rule='complete fixed 500-tick episodes; stop and report if realized overshoot/spread exceeds limits',
        seeds=asdict(replace(SeedBlocks(),validation=VALIDATION_SEEDS)),
        validation_energies=ENERGIES,assigned_trials_per_panel=len(VALIDATION_SEEDS)*len(ENERGIES),
        natural_episodes_per_panel=len(VALIDATION_SEEDS),
        assigned_followup='full dependency horizon, including parent follow-up after pup death; both dead is absorbing',
        assigned_geometry='parent four cardinal steps from nest; nearest nest food Chebyshev distance 4..8; parent sees food',
        reproduction_during_assigned_assay=False,final_test_executed=False,
        optimizer_cadence='one actor update per completed training episode; one separate critic update when enabled',
        guard_max_episodes_per_run=1000,torch_threads=2)


def run_pilot(output):
    """Exclusive output, protocol/source snapshot first, exactly eight approved runs."""
    output=Path(output).resolve()
    root=Path(__file__).resolve().parents[2]
    if output==root or root in output.parents:
        raise ValueError('pilot output must be outside the repository checkout')
    if output.exists():
        raise FileExistsError(output)
    if torch.get_num_threads()!=2:
        raise ValueError('set two CPU Torch threads before freezing the pilot')
    preservation=historical_integrity(root)
    output.mkdir(parents=True)
    spec=protocol()
    write_json(output/'protocol.json',spec)
    write_json(output/'historical-before.json',preservation)
    write_json(output/'provenance.json',dict(**provenance(),
        working_tree_status=subprocess.check_output(['git','-C',str(root),'status','--short'],text=True)))
    for name in provenance()['source_sha256']:
        target=output/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/'src'/name,target)
    for name in ('tests/test_v2.py','tests/test_v2_pilot.py','experiments/pilot_v2.py','requirements-reproduction.txt','docs/V2_PILOT_PROTOCOL.md'):
        target=output/'review_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target)
    initial_hashes, scenario_hashes, budgets = {}, {}, []
    runs=[]
    try:
        for algorithm,objective,seed in CELLS:
            name=f'{algorithm}-{objective}-s{seed}'
            directory=output/name
            directory.mkdir()
            learner=LearnerConfig(algorithm=algorithm,architecture='original',seed=seed,
                gamma=.99,actor_lr=.001,value_lr=.001,reward_scope='colony')
            exp=Experiment(training_config(),learner,RewardConfig(objective),
                replace(SeedBlocks(),validation=VALIDATION_SEEDS),validation_every=0,
                validation_decisions=INTERVAL,assigned_care=True,care_energies=ENERGIES)
            write_json(directory/'run-config.json',dict(environment=asdict(exp.cfg),
                learner=asdict(learner),rewards=asdict(exp.rewards),seeds=asdict(exp.seeds),
                actor_parameters=sum(p.numel() for p in exp.learner.policy.parameters()),
                critic_parameters=sum(p.numel() for p in exp.learner.value.parameters()) if exp.learner.value is not None else 0))
            initial_hash=state_hash(exp.learner.policy)
            if initial_hashes.setdefault(seed,initial_hash)!=initial_hash:
                raise RuntimeError('initial actor mismatch across algorithm/reward cells')
            initial=exp.validate_initial()
            def check_panel(panel):
                for row in panel['assigned']['trials']:
                    key=(row['world_seed'],row['initial_energy'])
                    if scenario_hashes.setdefault(key,row['initial_hash']) != row['initial_hash']:
                        raise RuntimeError('assigned initial worlds differ across methods/checkpoints')
            check_panel(initial)
            write_json(directory/'initial-validation.json',initial)
            initial_receipt=exp.checkpoint(directory/'initial.pt')
            checkpoints=[dict(path='initial.pt',sha256=initial_receipt,training_decisions=0,thresholds=[0])]
            print(f'START {name}: assigned {initial["assigned"]["summary"]["independent"]}/10, '
                  f'attendance {initial["assigned"]["summary"]["attendance_fraction"]:.3f}',flush=True)
            with (directory/'training.jsonl').open('x') as stream:
                while exp.training_decisions<TARGET:
                    if exp.episode_index>=1000:
                        raise RuntimeError('pilot episode guard reached')
                    row=exp.step()
                    if row['validation'] is not None:
                        check_panel(row['validation'])
                        digest=exp.checkpoint(directory/row['checkpoint_id'])
                        row['checkpoint_sha256']=digest
                        checkpoints.append(dict(path=row['checkpoint_id'],sha256=digest,
                            training_decisions=exp.training_decisions,thresholds=row['validation']['thresholds']))
                        score=row['validation']['assigned']['summary']
                        print(f'VALIDATE {name}: {exp.training_decisions} decisions, '
                              f'{exp.learner.updates} updates, assigned {score["independent"]}/10, '
                              f'attendance {score["attendance_fraction"]:.3f}',flush=True)
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                write_json(directory/'checkpoints.json',checkpoints)
            overshoot=exp.training_decisions-TARGET
            budgets.append(exp.training_decisions)
            if overshoot>MAX_OVERSHOOT or max(budgets)-min(budgets)>MAX_SPREAD:
                raise RuntimeError(f'budget tolerance exceeded: realized {budgets}; stop for approval')
            # Real checkpoint verification without additional training: restore optimizer,
            # scheduler and weights; replay one fixed assigned challenge exactly.
            final_path=directory/checkpoints[-1]['path']
            restored=Experiment.restore(final_path)
            if (state_hash(restored.learner.policy)!=state_hash(exp.learner.policy)
                or restored.next_validation_decision!=exp.next_validation_decision
                or restored.validation_history!=exp.validation_history):
                raise RuntimeError('final checkpoint restore mismatch')
            if exp.learner.value is not None and state_hash(restored.learner.value)!=state_hash(exp.learner.value):
                raise RuntimeError('critic checkpoint mismatch')
            replay=assigned_pup_trial(training_config(),restored.learner.policy,VALIDATION_SEEDS[0],ENERGIES[0])
            expected=exp.validation_history[-1]['assigned']['trials'][0]
            if replay!=expected:
                raise RuntimeError('final checkpoint assigned replay mismatch')
            summary=dict(name=name,algorithm=algorithm,objective=objective,seed=seed,
                target=TARGET,training_decisions=exp.training_decisions,overshoot=overshoot,
                optimizer_updates=exp.learner.updates,episodes=exp.episode_index,
                critic_optimizer_updates=exp.learner.updates if exp.learner.value is not None else 0,
                actor_parameters=sum(p.numel() for p in exp.learner.policy.parameters()),
                critic_parameters=sum(p.numel() for p in exp.learner.value.parameters()) if exp.learner.value is not None else 0,
                training_world_ticks=exp.training_ticks,validation_decisions=exp.validation_decisions,
                verification_replay_decisions=replay['adult_decisions'],initial_policy_hash=initial_hash,
                final_policy_hash=state_hash(exp.learner.policy),checkpoint_restore_and_assigned_replay=True,
                validations=exp.validation_history)
            write_json(directory/'summary.json',summary)
            runs.append(summary)
            plot_run(directory)
            print(f'COMPLETE {name}: {exp.training_decisions}, overshoot {overshoot}',flush=True)
        after=historical_integrity(root)
        if after!=preservation:
            raise RuntimeError('historical integrity mismatch')
        write_json(output/'historical-after.json',after)
        result=dict(protocol=spec,runs=runs,actual_budget_spread=max(budgets)-min(budgets),
            original_files_unchanged=len(after),historical_result_files_unchanged=sum(p.startswith('results/') for p in after),
            final_test_executed=False)
        write_json(output/'summary.json',result)
        export_tables(output,result)
        plot_overview(output,result)
        return result
    except Exception as exc:
        write_json(output/'STOPPED.json',dict(reason=str(exc),completed_runs=[r['name'] for r in runs],budgets=budgets))
        raise


def panel_rows(run):
    for panel in run['validations']:
        care=panel['assigned']['summary']
        natural=panel['summary']
        yield dict(run=run['name'],algorithm=run['algorithm'],objective=run['objective'],seed=run['seed'],
            thresholds=','.join(map(str,panel['thresholds'])),decisions=panel['actual_training_decisions'],
            updates=panel['optimizer_updates'],assigned_independent=care['independent'],assigned_trials=care['trials'],
            assigned_independence=care['independence_fraction'],assigned_attendance=care['attendance_fraction'],
            assigned_posthumous=care['posthumous_independence'],parent_deaths=care['parent_deaths'],
            natural_births=natural['births'],natural_eligible_births=natural['eligible_births'],
            natural_independence=natural['independence_fraction'],natural_attendance=natural['attendance_fraction'])


def export_tables(output,result):
    rows=[row for run in result['runs'] for row in panel_rows(run)]
    with (output/'validation-points.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    budgets=[{k:run[k] for k in ('name','training_decisions','overshoot','optimizer_updates','episodes',
              'critic_parameters','critic_optimizer_updates','validation_decisions')} for run in result['runs']]
    with (output/'budgets.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(budgets[0]));writer.writeheader();writer.writerows(budgets)


def load_rows(directory):
    return [json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines()]


def save_figure(fig,path):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    with path.open('xb') as stream:
        fig.savefig(stream,format='png',dpi=150)
    plt.close(fig)


def plot_run(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=json.loads((directory/'summary.json').read_text())
    points=list(panel_rows(run));rows=load_rows(directory)
    fig,axes=plt.subplots(3,3,figsize=(14,10))
    x=[p['decisions'] for p in points];train_x=[r['training_decisions'] for r in rows]
    def curve(ax,xx,yy,label):
        ax.plot(xx,[float('nan') if v is None else v for v in yy],marker='.',label=label)
        ax.set_title(label);ax.set_xlabel('Adult training decisions')
    curve(axes[0,0],x,[p['assigned_independence'] for p in points],'Assigned pup independence (10 trials)')
    curve(axes[0,1],x,[p['assigned_attendance'] for p in points],'Assigned attendance / care exposure')
    curve(axes[0,2],x,[p['natural_independence'] for p in points],'Natural eligible-offspring independence')
    curve(axes[1,0],train_x,[r['episode_metrics']['total_reward'] for r in rows],'Training reward (undiscounted)')
    curve(axes[1,1],train_x,[r['discounted_colony_return'] for r in rows],'Training discounted colony return')
    curve(axes[1,2],train_x,[r['entropy'] for r in rows],'Actor entropy')
    curve(axes[2,0],train_x,[r['gradient_norm'] for r in rows],'Actor gradient norm')
    curve(axes[2,1],train_x,[r['value_loss'] for r in rows],'Critic MC mean-squared error (N/A without critic)')
    curve(axes[2,2],train_x,[r['episode_metrics']['elapsed_ticks'] for r in rows],'Training episode duration')
    for ax in (axes[0,0],axes[0,1],axes[0,2]):
        ax.set_ylim(-.03,1.03)
    fig.suptitle(run['name']+' — descriptive controlled pilot')
    save_figure(fig,directory/'learning-curves.png')


def plot_overview(output,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,(algorithm,objective) in zip(axes.flat,itertools.product(('reinforce','reinforce_value'),('survival','offspring'))):
        for run in result['runs']:
            if run['algorithm']==algorithm and run['objective']==objective:
                points=list(panel_rows(run))
                ax.plot([p['decisions'] for p in points],[p['assigned_independence'] for p in points],
                        marker='o',label=f'seed {run["seed"]}')
        ax.set_title(f'{algorithm} / {objective}');ax.set_xlabel('Adult training decisions')
        ax.set_ylabel('Assigned-pup independence');ax.set_ylim(-.03,1.03);ax.legend()
    fig.suptitle('Controlled pilot — per-seed curves; no inferential intervals')
    save_figure(fig,output/'assigned-independence.png')
