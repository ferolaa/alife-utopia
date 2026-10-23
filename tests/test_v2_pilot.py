"""Protocol tests: no final-test worlds and no pilot training campaigns."""
from dataclasses import replace
import itertools
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agent import Agent
from brain import Brain
from config import DEFAULT, UNIVERSE_25
from v2.care import assigned_pup_trial, evaluate_assigned, summarize_assigned
from v2.data import Episode, Event, Transition, event_returns
from v2.environment import initialize
from v2.evaluation import SeedBlocks, state_hash
from v2.experiment import Experiment
from v2.learning import LearnerConfig, policy_loss
from v2.models import Policy

CFG=DEFAULT.variant('schedule-test',width=10,height=10,n_ticks=5,n_initial_agents=2,
                    n_food=0,food_respawn_rate=0,energy_cost_per_tick=0,
                    reproduce_threshold=1000,max_population=4)
BLOCKS=SeedBlocks(train_start=1000,train_size=100000,validation=(200100,),final_test=(300100,))


def test_coupled_two_agent_gradient_matches_independent_enumeration():
    # A acts at t=0, B at t=1; shared policy has state-dependent logits theta[0:2].
    # A receives .2*A at t=0. B receives 2*A + A*B + B at t=2.
    # A can thus improve B's reward without receiving that payment itself.
    gamma=.6
    theta=torch.tensor([-.3,.4],dtype=torch.float64,requires_grad=True)
    p,q=theta.sigmoid()
    expected=0.
    records=[]
    for a,b in itertools.product((0,1),repeat=2):
        probability=(p if a else 1-p)*(q if b else 1-q)
        expected=expected+probability*(.2*a+gamma**2*(2*a+a*b+b))
        ep=Episode(0,0,'two-agent-toy',3,
            [Transition(0,0,(0.,)*12,a,0.),Transition(1,1,(0.,)*12,b,0.)],
            [Event(0,0,.2*a,'early'),Event(2,1,2*a+a*b+b,'coupled')])
        records.append(ep)
    exact=torch.autograd.grad(expected,theta)[0]
    manual=torch.stack((p*(1-p)*(.2+gamma**2*(2+q)),gamma**2*q*(1-q)*(p+1))).detach()
    assert torch.allclose(exact,manual,atol=1e-12,rtol=0)
    def estimate(n,scope='colony',outer_discount=True):
        t=theta.detach().clone().requires_grad_()
        draws=torch.rand(n,2,generator=torch.Generator().manual_seed(111),dtype=torch.float64)
        actions=(draws < t.sigmoid().detach()).long()
        target_table=torch.tensor([event_returns(ep,gamma,scope) for ep in records],dtype=torch.float64)
        targets=target_table[2*actions[:,0]+actions[:,1]]
        logits=t.expand(n,2)
        logp=torch.where(actions.bool(),F.logsigmoid(logits),F.logsigmoid(-logits))
        discounts=torch.tensor([1.,gamma] if outer_discount else [1.,1.],dtype=torch.float64)
        loss=policy_loss(logp.reshape(-1),targets.reshape(-1),discounts.expand(n,2).reshape(-1),n)
        return -torch.autograd.grad(loss,t)[0]
    small=estimate(256)
    large=estimate(65536)
    assert torch.allclose(large,exact,atol=.006,rtol=0)
    assert torch.linalg.vector_norm(large-exact) < torch.linalg.vector_norm(small-exact)
    assert torch.linalg.vector_norm(estimate(65536,'individual_surrogate')-exact) > .15
    assert torch.linalg.vector_norm(estimate(65536,outer_discount=False)-exact) > .05


def test_decision_schedule_first_episode_boundary_and_initial_panel():
    exp=Experiment(CFG,seeds=BLOCKS,validation_every=0,validation_decisions=15)
    initial=exp.validate_initial()
    assert initial['thresholds']==[0] and initial['actual_training_decisions']==0
    assert exp.step()['validation'] is None
    second=exp.step()
    assert second['validation']['thresholds']==[15]
    assert second['validation']['actual_training_decisions']==20
    third=exp.step()
    assert third['validation']['thresholds']==[30]
    assert third['validation']['actual_training_decisions']==30
    assert exp.next_validation_decision==45
    with pytest.raises(ValueError):
        exp.validate_initial()


def test_multiple_crossed_thresholds_share_one_frozen_boundary():
    exp=Experiment(CFG,seeds=BLOCKS,validation_decisions=3)
    row=exp.step()
    assert row['validation']['thresholds']==[3,6,9]
    assert len(exp.validation_history)==1
    assert exp.next_validation_decision==12


@pytest.mark.parametrize('algorithm',('reinforce','reinforce_value'))
def test_decision_scheduled_validation_has_no_training_rng_leakage(algorithm):
    kwargs=dict(cfg=CFG,learner_config=LearnerConfig(algorithm=algorithm,seed=1),seeds=BLOCKS)
    scheduled=Experiment(**kwargs,validation_decisions=7)
    unscheduled=Experiment(**kwargs,validation_every=0)
    rng_before=torch.get_rng_state().clone()
    scheduled.validate_initial()
    for _ in range(3):
        scheduled.step();unscheduled.step()
    assert torch.equal(rng_before,torch.get_rng_state())
    assert state_hash(scheduled.learner.policy)==state_hash(unscheduled.learner.policy)
    if scheduled.learner.value is not None:
        assert state_hash(scheduled.learner.value)==state_hash(unscheduled.learner.value)
    assert scheduled.training_decisions==unscheduled.training_decisions==30
    assert scheduled.validation_decisions>0==unscheduled.validation_decisions


@pytest.mark.parametrize('algorithm',('reinforce','reinforce_value'))
def test_checkpoint_resume_preserves_pending_threshold_and_validation(tmp_path,algorithm):
    exp=Experiment(CFG,LearnerConfig(algorithm=algorithm),seeds=BLOCKS,validation_decisions=15)
    exp.validate_initial();exp.step();exp.step()
    path=tmp_path/'scheduled.pt';exp.checkpoint(path)
    restored=Experiment.restore(path)
    assert restored.next_validation_decision==30
    assert restored.validation_history==exp.validation_history
    a,b=exp.step(),restored.step()
    a.pop('training_seconds');b.pop('training_seconds')
    assert a==b
    assert state_hash(restored.learner.policy)==state_hash(exp.learner.policy)


def test_budget_overshoot_is_bounded_reported_and_enforced():
    exp=Experiment(CFG,seeds=BLOCKS,validation_every=0)
    assert exp.train_to_budget(15,5,max_overshoot=5)==dict(target=15,realized=20,overshoot=5)
    other=Experiment(CFG,seeds=BLOCKS,validation_every=0)
    with pytest.raises(RuntimeError,match='overshoot'):
        other.train_to_budget(15,5,max_overshoot=4)
    assert other.training_decisions==20  # Never erase or truncate already collected experience.


class Fixed(torch.nn.Module):
    def __init__(self,action=0):
        super().__init__();self.action=action
    def forward(self,x):
        logits=torch.full((len(x),6),-1000.)
        logits[:,self.action]=0
        return logits


def family(parent_lifespan=float('inf'),far=False,both_die=False):
    def make(cfg,seed):
        w,rng=initialize(cfg.variant('family',n_initial_agents=0,n_food=0,food_respawn_rate=0),seed)
        parent=Agent(5,5,50,Brain.random(rng),lifespan=parent_lifespan)
        pup=Agent(0 if far else 5,0 if far else 5,18,Brain.random(rng))
        pup.parent=parent;pup.dependent_until=cfg.dependency_ticks;parent.dependents=[pup]
        if both_die:
            parent.energy=0
        w.agents=[parent,pup]
        return w,rng
    return make


def care_config(**changes):
    return DEFAULT.variant('care-synthetic',n_food=0,food_respawn_rate=0,n_ticks=3,
                           dependency_ticks=3,parental_care_enabled=True,energy_cost_per_tick=0,**changes)


def test_assigned_independence_and_denominators():
    r=assigned_pup_trial(care_config(),Fixed(),0,20,initializer=family())
    assert r['independent'] and r['independence_tick']==3 and not r['posthumous_independence']
    assert r['outcome_followup_ticks']==r['simulated_ticks']==r['parent_at_risk_ticks']==3
    assert r['care_ticks']==r['attended_ticks']==2
    assert r['births']==0
    summary=summarize_assigned([r,r])
    assert summary['trials']==summary['independent']==2
    assert summary['care_ticks']==4 and summary['independence_fraction']==1
    assert 'eligible_births' not in summary


def test_assigned_posthumous_independence_and_full_parent_followup():
    r=assigned_pup_trial(care_config(),Fixed(),0,20,initializer=family(parent_lifespan=1))
    assert r['independent'] and r['posthumous_independence']
    assert r['parent_death_tick']==1 and r['independence_tick']==3
    assert r['parent_at_risk_ticks']==1 and r['outcome_followup_ticks']==3
    assert r['attended_ticks']==0 and r['care_ticks']==2


def test_pup_death_does_not_censor_parent_followup():
    r=assigned_pup_trial(care_config(neglect_tolerance=0),Fixed(),0,20,initializer=family(far=True))
    assert not r['independent'] and r['pup_death_tick']==1 and r['parent_alive']
    assert r['parent_at_risk_ticks']==r['outcome_followup_ticks']==r['simulated_ticks']==3
    assert r['pup_at_risk_ticks']==r['care_ticks']==1


def test_both_dead_absorbing_followup_is_explicit():
    r=assigned_pup_trial(care_config(neglect_tolerance=0),Fixed(),0,20,
                         initializer=family(parent_lifespan=1,far=True,both_die=True))
    assert not r['parent_alive'] and not r['pup_alive']
    assert r['simulated_ticks']==1 and r['absorbing_ticks']==2 and r['outcome_followup_ticks']==3


@pytest.mark.parametrize('architecture',('linear','original','deep'))
def test_standard_assigned_panel_frozen_replays_for_all_architectures(architecture):
    blocks=replace(SeedBlocks(),validation=(3_000_100,))
    policy=Policy(architecture,3)
    policy.train()
    before=state_hash(policy);rng_before=torch.get_rng_state().clone()
    first=evaluate_assigned(UNIVERSE_25,policy,blocks)
    second=evaluate_assigned(UNIVERSE_25,policy,blocks)
    assert first==second and state_hash(policy)==before and policy.training
    assert torch.equal(rng_before,torch.get_rng_state())
    assert first['summary']['trials']==2
    assert first['trials'][0]['layout_hash']==first['trials'][1]['layout_hash']
    assert first['trials'][0]['action_seed']==first['trials'][1]['action_seed']
    assert all(r['births']==0 and r['outcome_followup_ticks']==60 for r in first['trials'])


def test_standard_assigned_initial_worlds_match_across_methods():
    blocks=replace(SeedBlocks(),validation=(3_000_100,))
    a=evaluate_assigned(UNIVERSE_25,Policy('original',1),blocks)
    b=evaluate_assigned(UNIVERSE_25,Policy('original',2),blocks)
    assert [r['initial_conditions'] for r in a['trials']]==[r['initial_conditions'] for r in b['trials']]
    assert [r['initial_hash'] for r in a['trials']]==[r['initial_hash'] for r in b['trials']]


def test_assay_reproduction_actions_cannot_create_births():
    r=assigned_pup_trial(care_config(),Fixed(5),0,20,initializer=family())
    assert r['births']==0 and r['independent']
    assert r['action_counts']['reproduce']>0


def test_scheduled_assigned_validation_restore(tmp_path):
    cfg=UNIVERSE_25.variant('care-integration',n_ticks=2,n_initial_agents=2,max_population=8)
    exp=Experiment(cfg,seeds=replace(SeedBlocks(),validation=(3_000_100,)),
                   validation_decisions=1,assigned_care=True)
    exp.validate_initial();exp.step()
    path=tmp_path/'care.pt';exp.checkpoint(path)
    restored=Experiment.restore(path)
    assert restored.assigned_care and restored.care_energies==(10,20)
    assert restored.validation_history==exp.validation_history
    before=state_hash(exp.learner.policy)
    assert exp.validate([])==restored.validate([])
    assert state_hash(exp.learner.policy)==before


def test_pilot_protocol_is_exactly_the_approved_matrix():
    from v2.pilot import protocol, training_config
    spec=protocol()
    assert len(spec['cells'])==8
    assert {c['seed'] for c in spec['cells']}=={0,1}
    assert spec['architecture']=='original' and spec['gamma']==.99 and spec['reward_scope']=='colony'
    assert spec['target_training_decisions']==200000
    assert spec['validation_interval_decisions']==50000 and spec['max_overshoot_decisions']==20000
    assert spec['assigned_trials_per_panel']==10
    assert not spec['final_test_executed']
    cfg=training_config()
    assert not cfg.developmental_damage_enabled and not cfg.crowding_blocks_care and not cfg.nest_intrusion_enabled
    assert cfg.n_ticks==500 and cfg.n_initial_agents==40 and cfg.max_population==120


def test_pilot_refuses_existing_output_and_checkout(tmp_path):
    from v2.pilot import run_pilot
    with pytest.raises(FileExistsError):
        run_pilot(tmp_path)
    with pytest.raises(ValueError):
        run_pilot(Path(__file__).resolve().parents[1]/'results'/'forbidden')
