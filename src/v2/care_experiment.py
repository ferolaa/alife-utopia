"""Four complete on-policy care worlds per update, with separate validation.

This adapter deliberately does not change the historical full-colony Experiment.
Checkpoint every batch: complete frozen pre-update policies can be recovered.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from .care_task import CareTask, CONDITIONS, TASK_VERSION, TRAIN_START, evaluate_care
from .data import RewardConfig
from .evaluation import state_hash
from .experiment import provenance
from .learning import Learner,LearnerConfig

BATCH_EPISODES=4
# 100 parent actions + 41 pup actions (first at tick 59), per complete world.
MAX_EPISODE_DECISIONS=141
MAX_BATCH_DECISIONS=BATCH_EPISODES*MAX_EPISODE_DECISIONS


class CareExperiment:
    def __init__(self,learner_config=LearnerConfig(),objective='offspring'):
        if learner_config.seed not in (0,1) or learner_config.reward_scope!='colony' or learner_config.gamma!=.99:
            raise ValueError('care comparison requires seeds 0/1, colony credit and gamma .99')
        self.task=CareTask()
        self.learner=Learner(learner_config)
        self.rewards=RewardConfig(objective)
        self.episode_index=self.training_decisions=self.validation_decisions=0
        self.next_validation=5000
        self.history=[]
        self.validation_history=[]

    def batch(self,*,validate=True):
        if self.episode_index+BATCH_EPISODES>10000:
            raise ValueError('training replicate seed allocation exhausted')
        before=state_hash(self.learner.policy)
        started=time.perf_counter()
        episodes=[]
        for offset in range(BATCH_EPISODES):
            index=self.episode_index+offset
            seed=TRAIN_START+self.learner.config.seed*10000+index
            # Same condition ordering for both replicate seeds; algorithm and
            # objective never enter the world/action-seed or condition mapping.
            condition=CONDITIONS[index % len(CONDITIONS)]
            episodes.append(self.task.collect(self.learner.policy,seed,condition,
                objective=self.rewards.objective))
        if state_hash(self.learner.policy)!=before:
            raise RuntimeError('actor changed during batch collection')
        size=sum(e.decisions for e in episodes)
        if size>MAX_BATCH_DECISIONS or any(e.truncated for e in episodes):
            raise RuntimeError('invalid batch horizon/budget')
        update=self.learner.update(episodes)
        self.training_decisions+=size
        self.episode_index+=BATCH_EPISODES
        crossed=[]
        while self.training_decisions>=self.next_validation:
            crossed.append(self.next_validation)
            self.next_validation+=5000
        panel=self.validate(crossed) if validate and crossed else None
        # No destructive pop: archive full on-policy evidence for every batch.
        record=dict(batch=self.learner.updates,pre_update_actor_hash=before,
            post_update_actor_hash=state_hash(self.learner.policy),batch_decisions=size,
            training_decisions=self.training_decisions,episodes=self.episode_index,
            worlds=[dict(world_seed=e.world_seed,action_seed=e.action_seed,
                initial_hash=e.initial_hash,horizon=e.horizon,metrics=e.metrics,
                transitions=[asdict(t) for t in e.transitions],events=[asdict(e) for e in e.events])
                for e in episodes],diagnostics=update,validation=panel,
            training_and_validation_seconds=time.perf_counter()-started)
        record=json.loads(json.dumps(record,allow_nan=False))
        self.history.append(record)
        return record

    def validate_initial(self):
        if self.episode_index or self.validation_history:
            raise ValueError('initial validation must run once before training')
        return self.validate([0])

    def validate(self,thresholds=()):
        panel=evaluate_care(self.task,self.learner.policy)
        self.validation_decisions+=panel['summary']['adult_decisions']
        row=dict(thresholds=list(thresholds),actual_training_decisions=self.training_decisions,
            optimizer_updates=self.learner.updates,cumulative_validation_decisions=self.validation_decisions,
            panel=panel)
        row=json.loads(json.dumps(row,allow_nan=False))
        self.validation_history.append(row)
        return row

    def train_to_budget(self,output):
        """Future single-cell execution API; not called by the review CLI.

        The budget is fixed at 20,000, never increased to obtain a success.
        Source snapshots and every batch boundary are preserved for diagnosis.
        """
        if self.episode_index or self.validation_history:
            raise ValueError('budget runner requires a fresh experiment')
        if self.learner.config.architecture!='original':
            raise ValueError('proposed scientific pilot uses original architecture only')
        output=Path(output).resolve()
        root=Path(__file__).resolve().parents[2]
        if output==root or root in output.parents:
            raise ValueError('pilot outputs must be outside checkout')
        output.mkdir(parents=True,exist_ok=False)
        def write(name,value):
            with (output/name).open('x') as stream:
                json.dump(value,stream,indent=2,allow_nan=False)
                stream.write('\n')
        metadata=provenance()
        sources=output/'source';sources.mkdir()
        for name in metadata['source_sha256']:
            destination=sources/name;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes((root/'src'/name).read_bytes())
        write('protocol.json',dict(**protocol(),cell=asdict(self.learner.config),
            objective=self.rewards.objective,provenance=metadata))
        checkpoints=[]
        def save():
            name=f'update-{self.learner.updates:04d}.pt'
            checkpoints.append(dict(path=name,sha256=self.checkpoint(output/name),
                training_decisions=self.training_decisions,updates=self.learner.updates))
        write('initial-validation.json',self.validate_initial());save()
        with (output/'training.jsonl').open('x') as log:
            for _ in range(1000):
                row=self.batch()
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();save()
                if self.training_decisions>=20000:
                    overshoot=self.training_decisions-20000
                    if overshoot>=MAX_BATCH_DECISIONS:
                        raise RuntimeError('prespecified batch overshoot exceeded')
                    write('checkpoints.json',checkpoints)
                    result=dict(target=20000,realized=self.training_decisions,overshoot=overshoot,
                        episodes=self.episode_index,optimizer_updates=self.learner.updates,
                        validation_decisions=self.validation_decisions,final_test_executed=False)
                    write('summary.json',result)
                    return result
        raise RuntimeError('batch guard exhausted; stop, do not extend budget')

    def checkpoint(self,path):
        path=Path(path)
        payload=dict(schema='care-experiment-1',task_version=TASK_VERSION,provenance=provenance(),
            followup_ticks=self.task.followup_ticks,learner=self.learner.state(),
            reward=asdict(self.rewards),episode_index=self.episode_index,
            training_decisions=self.training_decisions,validation_decisions=self.validation_decisions,
            # Transition observations originate in NumPy. Keep audit metadata
            # primitive-only so safe weights_only restoration needs no globals.
            next_validation=self.next_validation,
            history=json.loads(json.dumps(self.history,allow_nan=False)),
            validation_history=json.loads(json.dumps(self.validation_history,allow_nan=False)))
        with path.open('xb') as stream:
            torch.save(payload,stream)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def restore(cls,path,expected_sha256):
        path=Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
            raise ValueError('checkpoint checksum mismatch')
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['schema']!='care-experiment-1' or data['task_version']!=TASK_VERSION or data['followup_ticks']!=40:
            raise ValueError('checkpoint task/schema mismatch')
        current=provenance()
        for key in ('python','torch','numpy','source_sha256','device','torch_threads'):
            if current[key]!=data['provenance'][key]:
                raise ValueError(f'checkpoint runtime/source mismatch: {key}')
        obj=cls(LearnerConfig(**data['learner']['config']),data['reward']['objective'])
        if asdict(obj.rewards)!=data['reward']:
            raise ValueError('checkpoint reward mismatch')
        obj.learner=Learner.restore(data['learner'])
        for key in ('episode_index','training_decisions','validation_decisions','next_validation','history','validation_history'):
            setattr(obj,key,data[key])
        return obj


def protocol():
    return dict(status='PROPOSED: scientific execution not authorized',task_version=TASK_VERSION,
        cells=[dict(algorithm=a,objective=o,seed=s) for a in ('reinforce','reinforce_value')
               for o in ('survival','offspring') for s in (0,1)],architecture='original',
        reward_scope='colony',gamma=.99,survival_reward=.01,independence_reward=1.,
        actor_lr=.001,value_lr=.001,entropy_coefficient=0.,gradient_clipping=None,
        return_normalization=False,episodes_per_update=BATCH_EPISODES,
        target_training_decisions=20000,validation_interval_decisions=5000,
        max_episode_decisions=MAX_EPISODE_DECISIONS,max_batch_decisions=MAX_BATCH_DECISIONS,
        max_overshoot_decisions=MAX_BATCH_DECISIONS-1,
        max_between_run_decision_spread=MAX_BATCH_DECISIONS-1,
        training_world_seed_rule='2100000 + 10000 * replicate_seed + episode_index',
        condition_rule='CONDITIONS[episode_index % 18], same order for all cells',
        validation_trials_per_panel=72,validation_points=[0,5000,10000,15000,20000],
        final_test_executed=False,minimum_updates_for_20000_decisions=36,
        budget_extension='requires separate approval; never automatic',
        tuning_budget_per_cell=0,checkpoint_schedule='initial and every completed update, exclusive paths',
        stop_rule='first complete four-world batch reaching 20000 adult decisions; no early success stopping',
        comparison='algorithm/objective paired within replicate; two replicates descriptive only')
