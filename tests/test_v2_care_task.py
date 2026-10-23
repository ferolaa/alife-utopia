"""Small, bounded regression and mathematical tests, not a scientific campaign."""
from dataclasses import asdict
import hashlib
import random
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agent import Agent
from brain import Brain
from config import UNIVERSE_25
from world import World
from v2.care_task import (CareTask,CareCondition,CareControl,CONDITIONS,VALIDATION_SEEDS,
    evaluate_care,checked_seed,summarize_care)
from v2.care_experiment import CareExperiment,protocol,MAX_BATCH_DECISIONS
from v2.data import event_returns
from v2.evaluation import state_hash
from v2.learning import Learner,LearnerConfig,policy_loss
from v2.models import Policy


class Fixed(torch.nn.Module):
    def __init__(self,action=0):super().__init__();self.action=action
    def forward(self,x):
        y=torch.full((len(x),6),-1000.);y[:,self.action]=0.;return y


def synthetic(task,monkeypatch,*,parent_lifespan=1000,pup_age=0,pup_energy=18,
              pup_lifespan=1000,parent_energy=50,neglect=0,far=False):
    def initialize(seed,condition,split):
        cfg=task.cfg.variant('synthetic',width=24,height=24,n_food=0,
            food_respawn_rate=0,energy_cost_per_tick=.25,crowding_cost_enabled=False)
        rng=random.Random(seed)
        w=World(24,24,0,rng=rng)
        parent=Agent(14 if far else 10,10,parent_energy,Brain.random(rng),parent_lifespan)
        parent.age=120
        pup=Agent(10,10,pup_energy,Brain.random(rng),pup_lifespan)
        pup.age=pup_age;pup.parent=parent;pup.dependent_until=60;pup.nest=(10,10)
        pup.neglect_ticks=neglect
        parent.dependents=[pup]
        w.nests.add((10,10));assert w.claim_nest(10,10)
        w.agents=[parent,pup]
        return cfg,w,rng
    monkeypatch.setattr(task,'initialize',initialize)


@pytest.mark.parametrize('condition',CONDITIONS)
def test_initialization_is_a_real_family_with_unchanged_body_rules(condition):
    task=CareTask();cfg,w,r=task.initialize(9_000_001,condition,'technical')
    parent,pup=w.agents
    assert parent.age==120 and pup.age==0 and parent.energy==condition.parent_energy
    assert pup.parent is parent and parent.dependents==[pup] and pup.is_dependent
    assert parent.children==1
    assert pup.nest in w.occupied_nests and pup.dependent_until==60 and pup.energy==18
    assert max(abs(parent.x-pup.x),abs(parent.y-pup.y))==condition.parent_distance
    assert min(max(abs(fx-pup.x),abs(fy-pup.y)) for fx,fy in w.food)>=condition.food_distance
    for key in ('energy_max','energy_cost_per_tick','energy_from_food','dependency_ticks',
                'care_radius','neglect_tolerance','maturity_age','ageing_enabled',
                'max_age','max_age_spread','crowding_cost_enabled','crowding_energy_cost'):
        assert getattr(cfg,key)==getattr(UNIVERSE_25,key)
    assert cfg.reproduce_threshold==float('inf') and UNIVERSE_25.reproduce_threshold==35


def test_collected_observation_and_action_use_actual_world_sensing():
    task=CareTask();c=CareCondition(4,12,4)
    cfg,w,_=task.initialize(9_000_001,c,'technical');w.rebuild_occupancy(cfg.crowding_radius)
    expected=w.agents[0].sense(w,cfg)
    ep=task.collect(Fixed(0),9_000_001,c,split='technical')
    assert ep.transitions[0].observation==expected and ep.transitions[0].action==0
    assert expected[2]>0 and expected[11]>0 and len(expected)==12
    # Neural inputs precede Agent.act's age/cost processing, as in the old collector.
    assert expected[3]==12/60 and ep.transitions[0].tick==0


def test_natural_maturation_reward_clock_and_post_independence_followup(monkeypatch):
    task=CareTask();synthetic(task,monkeypatch)
    ep=task.collect(Fixed(),9_000_001,split='technical')
    events=[e for e in ep.events if e.kind=='independence']
    assert len(events)==1 and (events[0].tick,events[0].recipient,events[0].subject)==(59,0,1)
    assert ep.metrics['independence_tick']==60 and ep.metrics['parent_alive_at_independence']
    assert ep.metrics['care_ticks']==ep.metrics['attended_ticks']==59
    assert ep.metrics['pup_adult_decisions']==41 and ep.metrics['parent_adult_decisions']==100
    assert ep.decisions==141 and ep.metrics['elapsed_ticks']==ep.horizon==100
    assert ep.metrics['outcome_followup_ticks']==100 and not ep.truncated
    assert event_returns(ep,.99)[0]==pytest.approx(sum(.99**e.tick*e.amount for e in ep.events))
    assert all(e.amount==(.01 if e.kind=='survival' else 1) for e in ep.events)


def test_posthumous_credit_continues_after_parent_trajectory_ends(monkeypatch):
    task=CareTask();synthetic(task,monkeypatch,parent_lifespan=160)
    ep=task.collect(Fixed(),9_000_001,split='technical')
    m=ep.metrics
    assert m['parent_death_tick']==40 and m['independence_tick']==60
    assert m['posthumous_independence'] and not m['parent_alive_at_independence']
    parent=[t for t in ep.transitions if t.agent==0]
    assert len(parent)==40 and parent[-1].terminated
    own=event_returns(ep,.99,'individual_surrogate')
    assert own[0]==pytest.approx(sum(.01*.99**t for t in range(39))+.99**59)
    assert m['elapsed_ticks']==100


def test_maturation_tick_death_never_earns_independence(monkeypatch):
    task=CareTask();synthetic(task,monkeypatch,pup_energy=0)
    ep=task.collect(Fixed(),9_000_001,split='technical')
    assert ep.metrics['pup_death_tick']==60 and not ep.metrics['independent']
    assert not any(e.kind=='independence' for e in ep.events)
    assert ep.metrics['elapsed_ticks']==100  # surviving parent is still followed


def test_pup_death_does_not_end_parent_followup_and_extinction_is_absorbing(monkeypatch):
    task=CareTask();synthetic(task,monkeypatch,far=True)
    ep=task.collect(Fixed(),9_000_001,split='technical')
    assert ep.metrics['pup_death_tick']==26 and ep.metrics['parent_at_risk_ticks']==100
    assert ep.metrics['outcome_followup_ticks']==100 and ep.metrics['care_ticks']==26
    synthetic(task,monkeypatch,far=True,parent_energy=.1)
    dead=task.collect(Fixed(),9_000_001,split='technical')
    assert dead.metrics['elapsed_ticks']==26 and dead.metrics['absorbing_ticks']==74
    assert dead.metrics['outcome_followup_ticks']==100 and not dead.truncated


def test_external_truncation_is_rejected_by_mc(monkeypatch):
    task=CareTask();synthetic(task,monkeypatch)
    ep=task.collect(Policy(),9_000_001,split='technical',max_ticks=20)
    assert ep.truncated and not ep.metrics['independent']
    assert ep.metrics['outcome_followup_ticks']==20 and ep.metrics['absorbing_ticks']==0
    with pytest.raises(ValueError,match='completed'):
        Learner().update([ep])


def test_reproduction_blocked_but_never_rewarded_for_inactivity():
    task=CareTask();ep=task.collect(Fixed(5),9_000_001,CareCondition(4,6,0),split='technical')
    assert ep.metrics['births']==ep.metrics['cap_rejections']==0
    assert ep.metrics['action_counts']['reproduce']>0 and not ep.metrics['independent']
    assert all(e.kind=='survival' for e in ep.events)


def test_frozen_panel_rng_isolation_and_matched_worlds():
    task=CareTask();p=Policy();p.train();before=state_hash(p)
    py=random.getstate();np_before=np.random.get_state();tr=torch.get_rng_state().clone()
    kwargs=dict(seeds=VALIDATION_SEEDS[:1],conditions=CONDITIONS[:2])
    a=evaluate_care(task,p,**kwargs);b=evaluate_care(task,p,**kwargs)
    c=evaluate_care(task,Policy(seed=1),**kwargs)
    assert a==b and p.training and state_hash(p)==before
    assert [r['initial_conditions'] for r in a['trials']]==[r['initial_conditions'] for r in c['trials']]
    assert py==random.getstate() and np.array_equal(np_before[1],np.random.get_state()[1])
    assert torch.equal(tr,torch.get_rng_state())
    with pytest.raises(ValueError):evaluate_care(task,p,seeds=(4_000_000,))


@pytest.mark.parametrize('seed,split',[(4_000_000,'train'),(4_000_010,'validation'),
    (3_000_300,'train'),(2_100_000,'validation'),(9_000_000,'final_test')])
def test_split_access_rejects_final_and_leakage(seed,split):
    with pytest.raises(ValueError):checked_seed(seed,split)


def test_real_task_objectives_differ_only_in_bonus_events():
    t=CareTask();p=Policy(seed=0);c=CareCondition(4,20,0)
    a=t.collect(p,2_100_000,c,objective='survival')
    b=t.collect(p,2_100_000,c,objective='offspring')
    assert a.initial_hash==b.initial_hash and a.transitions==b.transitions
    assert [e for e in b.events if e.kind=='survival']==a.events
    assert a.metrics['independent']==b.metrics['independent']


def test_real_care_actions_match_exact_enumerated_policy_gradient(monkeypatch):
    # A near-neglect pup matures next tick. First parent action (toward pup or
    # stay) determines whether it survives. All later actions are held fixed.
    task=CareTask();synthetic(task,monkeypatch,pup_age=58,neglect=25,far=True)
    class FirstThenStay(Fixed):
        def __init__(self,first):super().__init__(0);self.first=first;self.calls=0
        def forward(self,x):
            self.calls+=1
            return Fixed(self.first if self.calls==1 else 0)(x)
    bad=task.collect(FirstThenStay(0),9_000_001,split='technical')
    good=task.collect(FirstThenStay(4),9_000_001,split='technical')
    assert not bad.metrics['independent'] and good.metrics['independent']
    gamma=.99;theta=torch.tensor(.3,dtype=torch.float64,requires_grad=True);p=theta.sigmoid()
    J=torch.tensor([sum(gamma**e.tick*e.amount for e in ep.events) for ep in (bad,good)],dtype=torch.float64)
    exact=torch.autograd.grad((1-p)*J[0]+p*J[1],theta,retain_graph=True)[0]
    G=torch.tensor([event_returns(ep,gamma)[0] for ep in (bad,good)],dtype=torch.float64)
    probability=torch.stack((1-p,p)).detach()
    logp=torch.stack((torch.log1p(-p),p.log()))
    for baseline in (0.,5.):
        loss=policy_loss(logp,probability*(G-baseline),torch.ones(2,dtype=torch.float64))
        estimated=-torch.autograd.grad(loss,theta,retain_graph=True)[0]
        assert torch.allclose(estimated,exact,atol=1e-12,rtol=0)
    assert exact>0
    # The old own-agent surrogate misses survival of the independently acting pup.
    own=torch.tensor([event_returns(ep,gamma,'individual_surrogate')[0] for ep in (bad,good)],dtype=torch.float64)
    assert own[1]<G[1] and good.events[-1].tick==99


@pytest.mark.parametrize('algorithm',('reinforce','reinforce_value'))
@pytest.mark.parametrize('architecture',('linear','original','deep'))
def test_four_world_update_gradient_and_resume(tmp_path,algorithm,architecture,monkeypatch):
    config=LearnerConfig(algorithm=algorithm,architecture=architecture)
    exp=CareExperiment(config);exp.batch(validate=False)
    assert exp.learner.updates==1 and exp.episode_index==4
    assert exp.training_decisions==sum(r['metrics']['adult_decisions'] for r in exp.history[0]['worlds'])
    assert exp.training_decisions<=MAX_BATCH_DECISIONS
    path=tmp_path/'one.pt';checksum=exp.checkpoint(path)
    restored=CareExperiment.restore(path,checksum)
    a=exp.batch(validate=False);b=restored.batch(validate=False)
    a.pop('training_and_validation_seconds');b.pop('training_and_validation_seconds')
    assert a==b and state_hash(exp.learner.policy)==state_hash(restored.learner.policy)
    if exp.learner.value is not None:assert state_hash(exp.learner.value)==state_hash(restored.learner.value)
    with pytest.raises(FileExistsError):exp.checkpoint(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==checksum
    bad=tmp_path/'bad.pt';bad.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError,match='checksum'):CareExperiment.restore(bad,checksum)


@pytest.mark.parametrize('algorithm',('reinforce','reinforce_value'))
def test_batch_actor_gradient_matches_manual_four_world_mean(algorithm,monkeypatch):
    learner=Learner(LearnerConfig(algorithm=algorithm));task=CareTask()
    episodes=[task.collect(learner.policy,2_100_000+i,CONDITIONS[i]) for i in range(4)]
    records=[t for e in episodes for t in e.transitions]
    x=torch.tensor([t.observation for t in records],dtype=torch.float32)
    actions=torch.tensor([t.action for t in records])
    target=torch.tensor([r for e in episodes for r in event_returns(e,.99)])
    discounts=torch.tensor([.99**t.tick for t in records])
    baseline=learner.value(x).detach() if learner.value is not None else torch.zeros_like(target)
    loss=-(Categorical(logits=learner.policy(x)).log_prob(actions)*(target-baseline)*discounts).sum()/4
    manual=torch.autograd.grad(loss,tuple(learner.policy.parameters()))
    monkeypatch.setattr(learner.optimizer,'step',lambda:None)
    learner.update(episodes)
    assert all(torch.allclose(p.grad,g,atol=1e-6,rtol=1e-5) for p,g in zip(learner.policy.parameters(),manual))


def test_controls_produce_different_outcomes_and_feeding_has_value():
    task=CareTask();kwargs=dict(seeds=VALIDATION_SEEDS[:1],conditions=CONDITIONS[:6])
    care=evaluate_care(task,CareControl(),**kwargs)['summary']
    abandon=evaluate_care(task,CareControl('abandonment'),**kwargs)['summary']
    forage=evaluate_care(task,CareControl('forage_care'),**kwargs)['summary']
    assert care['independent']>abandon['independent']
    assert care['attendance_fraction']>abandon['attendance_fraction']+.3
    assert forage['parent_alive_at_independence']>care['parent_alive_at_independence']
    assert forage['parent_survival_fraction']>care['parent_survival_fraction']
    assert forage['parent_feeds']>0==care['parent_feeds']
    low=evaluate_care(task,CareControl(),seeds=VALIDATION_SEEDS[:1],conditions=CONDITIONS[:2])['summary']
    assert low['independent']==0  # cannot pass the hardest energy stratum by sitting


def test_protocol_is_proposed_eight_run_matrix_with_bounded_complete_batches():
    p=protocol()
    assert len(p['cells'])==8 and p['target_training_decisions']==20000
    assert p['episodes_per_update']==4 and p['validation_trials_per_panel']==72
    assert p['max_overshoot_decisions']==563 and p['minimum_updates_for_20000_decisions']==36
    assert p['architecture']=='original' and not p['final_test_executed']
    assert p['tuning_budget_per_cell']==0


def test_feeding_observation_does_not_change_physics_or_sampling():
    from v2.environment import collect
    from v2.data import RewardConfig
    task=CareTask();policy=CareControl('forage_care');c=CareCondition(4,6,4)
    cfg,w,rng=task.initialize(9_000_001,c,'technical')
    plain=collect(cfg,policy,9_000_001,RewardConfig('offspring'),initial=lambda c,s:(w,rng))
    observed=task.collect(policy,9_000_001,c,split='technical')
    assert plain.transitions==observed.transitions and plain.events==observed.events
    for key,value in plain.metrics.items():assert observed.metrics[key]==value
    assert observed.metrics['parent_feeds']>0


@pytest.mark.parametrize('algorithm',('reinforce','reinforce_value'))
def test_validation_changes_no_training_state_or_future_updates(algorithm,monkeypatch):
    import v2.care_experiment as module
    original=module.evaluate_care
    monkeypatch.setattr(module,'evaluate_care',lambda task,p:original(task,p,
        seeds=VALIDATION_SEEDS[:1],conditions=CONDITIONS[:2]))
    a=CareExperiment(LearnerConfig(algorithm=algorithm))
    b=CareExperiment(LearnerConfig(algorithm=algorithm))
    panel=a.validate_initial()
    assert panel['thresholds']==[0] and panel['actual_training_decisions']==0
    assert a.validation_decisions>0 and a.training_decisions==a.learner.updates==0
    with pytest.raises(ValueError):a.validate_initial()
    x=a.batch(validate=False);y=b.batch(validate=False)
    x.pop('training_and_validation_seconds');y.pop('training_and_validation_seconds')
    assert x==y and state_hash(a.learner.policy)==state_hash(b.learner.policy)
    if a.learner.value is not None:assert state_hash(a.learner.value)==state_hash(b.learner.value)


def test_validation_occurs_at_first_complete_batch_boundary(monkeypatch):
    exp=CareExperiment();exp.training_decisions=4990
    seen=[]
    def validate(thresholds):
        seen.append((list(thresholds),exp.training_decisions,exp.episode_index,exp.learner.updates))
        return dict(thresholds=thresholds,actual_training_decisions=exp.training_decisions)
    monkeypatch.setattr(exp,'validate',validate)
    record=exp.batch()
    assert len(seen)==1 and seen[0][0]==[5000]
    assert seen[0][2:]==(4,1) and seen[0][1]==record['training_decisions']
    assert 5000<=record['training_decisions']<5000+MAX_BATCH_DECISIONS
    assert exp.next_validation==10000


def test_fixed_budget_runner_stops_at_first_complete_batch_without_training(tmp_path,monkeypatch):
    exp=CareExperiment();seen=[]
    monkeypatch.setattr(exp,'validate_initial',lambda:dict(technical_fake_panel=True))
    monkeypatch.setattr(exp,'checkpoint',lambda p:'technical-fake-checksum')
    def batch():
        seen.append(1);exp.training_decisions+=564;exp.episode_index+=4;exp.learner.updates+=1
        return dict(technical_fake_batch=True)
    monkeypatch.setattr(exp,'batch',batch)
    result=exp.train_to_budget(tmp_path/'mock-budget')
    assert len(seen)==36 and result['realized']==20304 and result['overshoot']==304
    assert result['episodes']==144 and result['optimizer_updates']==36
    assert result['validation_decisions']==0  # mock used no real evaluation/training
    with pytest.raises(ValueError):exp.train_to_budget(tmp_path/'again')
