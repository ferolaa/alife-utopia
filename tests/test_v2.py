"""Mathematical and small-world tests; never opens final research seed worlds."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from agent import Agent
from brain import ACTIONS, Brain
from config import DEFAULT
from rl import PolicyNet, run_episode
from v2.data import Episode, Event, RewardConfig, Transition, event_returns
from v2.environment import collect, initialize
from v2.evaluation import SeedBlocks, evaluate, state_hash, summarize
from v2.experiment import Experiment, run_smoke
from v2.learning import Learner, LearnerConfig, discounted_returns, policy_loss
from v2.models import HandDesigned, Policy, Uniform, load_legacy
from v2.plotting import learning_curves

CFG = DEFAULT.variant('tiny', width=10, height=10, n_ticks=6, n_initial_agents=2,
                      n_food=10, max_population=8, parental_care_enabled=True, dependency_ticks=2)
# Test-only world panel, independent of the reserved scientific final-test block.
BLOCKS = SeedBlocks(train_start=1000, train_size=100000, validation=(200001, 200002), final_test=(300001,))


@pytest.mark.parametrize('architecture,count', [('linear',78), ('original',234), ('deep',5382)])
def test_shapes_probabilities_parameter_counts_and_sampling(architecture,count):
    policy = Policy(architecture, 7)
    x = torch.randn(5,12)
    assert policy(x).shape == (5,6)
    assert sum(p.numel() for p in policy.parameters()) == count
    probs = policy.probabilities(x)
    assert torch.all(probs >= 0) and torch.allclose(probs.sum(-1), torch.ones(5))
    g = torch.Generator().manual_seed(4)
    first = policy.sample(x,g)
    assert first.shape == (5,) and first.min() >= 0 and first.max() < 6
    assert torch.equal(first, policy.sample(x,torch.Generator().manual_seed(4)))


def test_uniform_sample_frequencies():
    policy = Policy('linear')
    with torch.no_grad():
        for p in policy.parameters():
            p.zero_()
    actions = policy.sample(torch.zeros(12000,12), torch.Generator().manual_seed(0))
    counts = torch.bincount(actions,minlength=6)/12000
    assert torch.all((counts-1/6).abs() < .02)


@pytest.mark.parametrize('action',range(6))
def test_reinforce_gradient_direction(action):
    logits = torch.zeros(6,requires_grad=True)
    loss = policy_loss(logits.log_softmax(0)[action:action+1],torch.tensor([2.]),torch.ones(1))
    loss.backward()
    assert logits.grad[action] < 0
    assert torch.all(logits.grad[torch.arange(6)!=action] > 0)
    assert torch.allclose(logits.grad.sum(),torch.tensor(0.),atol=1e-6)


def test_advantage_detached_and_baseline_direction():
    logp = torch.tensor([-.5,-.5],requires_grad=True)
    advantage = torch.tensor([1.,-1.],requires_grad=True)
    policy_loss(logp,advantage,torch.ones(2)).backward()
    assert torch.equal(logp.grad,torch.tensor([-1.,1.]))
    assert advantage.grad is None


def test_analytic_expected_bandit_gradient():
    # Exact expectation over actions: derivative of -E[reward] at uniform logits.
    logits = torch.zeros(2,requires_grad=True)
    lp = logits.log_softmax(0)
    weights = lp.exp().detach()
    policy_loss(lp,torch.tensor([0.,1.])*weights,torch.ones(2)).backward()
    assert torch.allclose(logits.grad,torch.tensor([.25,-.25]))


def test_return_boundaries():
    assert discounted_returns([1,2,3],.5,terminated=True) == [2.75,3.5,3.]
    assert discounted_returns([1,2],.5,truncated=True,bootstrap=8) == [4.,6.]
    with pytest.raises(ValueError):
        discounted_returns([1],.99,truncated=True)
    with pytest.raises(ValueError):
        discounted_returns([1],.99,terminated=True,truncated=True)


def trajectory(tick=0,agent=0):
    return Transition(tick,agent,(0.,)*12,0,0.)


def test_delayed_and_posthumous_credit_uses_world_clock():
    episode = Episode(0,0,'synthetic',6,[trajectory(0),trajectory(2),trajectory(3,1)],
                      [Event(5,0,1.,'independence',2)])
    assert event_returns(episode,.5,'individual_surrogate') == [1/32,1/8,0.]
    assert event_returns(episode,.5,'colony') == [1/32,1/8,1/4]
    assert event_returns(episode,1.) == [1.,1.,1.]
    episode.truncated = True
    with pytest.raises(ValueError):
        event_returns(episode,.99)


def test_gamma_zero_and_multiple_offspring_events():
    ep = Episode(0,0,'x',2,[trajectory()], [Event(0,0,1.,'independence',1),
                                          Event(0,0,1.,'independence',2),Event(1,0,5.,'survival')])
    assert event_returns(ep,0.) == [2.]
    assert event_returns(ep,1.) == [7.]
    assert event_returns(Episode(0,0,'x',2,[trajectory()]),.99) == [0.]


class Fixed(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action=action
    def forward(self,x):
        scores=torch.full((len(x),6),-1000.)
        scores[:,self.action]=0.
        return scores


def family(parent_lifespan=float('inf'), pup_ages=(0,), far=False):
    def create(cfg,seed):
        world,rng=initialize(cfg.variant('family',n_initial_agents=0,n_food=0),seed)
        parent=Agent(5,5,50,Brain.random(rng),lifespan=parent_lifespan)
        pups=[]
        for age in pup_ages:
            pup=Agent(0 if far else 5,0 if far else 5,18,Brain.random(rng))
            pup.parent=parent; pup.dependent_until=cfg.dependency_ticks; pup.age=age
            pups.append(pup)
        parent.dependents=pups.copy()
        world.agents=[parent,*pups]
        return world,rng
    return create


@pytest.mark.parametrize('objective',('survival','offspring'))
def test_multiple_dependents_metrics_and_rewards(objective):
    ep=collect(CFG.variant('family',n_ticks=3,energy_cost_per_tick=0),Fixed(0),1,
               RewardConfig(objective),initial=family(pup_ages=(0,0)))
    assert ep.metrics['independent_offspring']==2
    assert ep.metrics['care_ticks']==ep.metrics['attended_ticks']==2
    assert ep.metrics['adult_decisions']==7
    assert len([e for e in ep.events if e.kind=='independence'])==(2 if objective=='offspring' else 0)
    assert ep.metrics['starting_population']==3


def test_parent_death_before_independence_retains_timestamp():
    ep=collect(CFG.variant('death',n_ticks=4,energy_cost_per_tick=0,neglect_tolerance=10),
               Fixed(0),1,RewardConfig('offspring'),initial=family(parent_lifespan=1))
    events=[e for e in ep.events if e.kind=='independence']
    assert len(events)==1 and events[0].tick==1 and events[0].recipient==0
    parent=[t for t in ep.transitions if t.agent==0]
    assert len(parent)==1 and parent[0].terminated
    assert event_returns(ep,.5,'individual_surrogate')[0]==.5
    assert ep.metrics['attended_ticks']==0 and ep.metrics['care_ticks']==1


def test_zero_dependents_undefined_care_denominator():
    ep=collect(CFG,Fixed(0),0,initial=family(pup_ages=()))
    assert ep.metrics['attendance_fraction'] is None
    assert ep.metrics['independence_fraction'] is None
    assert ep.metrics['independent_offspring']==0


def test_neglect_death_is_observed_once():
    ep=collect(CFG.variant('neglect',n_ticks=3,neglect_tolerance=0),Fixed(0),1,
               RewardConfig('offspring'),initial=family(far=True))
    assert ep.metrics['deaths_by_cause']['neglect']==1
    assert ep.metrics['independent_offspring']==0
    assert ep.metrics['care_ticks']==1 and ep.metrics['attendance_fraction']==0


def test_extinction_terminal_vs_external_truncation():
    death=CFG.variant('dead',n_food=0,energy_start=.1,energy_cost_per_tick=1)
    ep=collect(death,Fixed(0),1)
    assert ep.metrics['extinct'] and not ep.truncated
    assert ep.metrics['elapsed_ticks']==1
    assert all(t.terminated and not t.truncated for t in ep.transitions)
    external=collect(CFG,Fixed(0),1,max_ticks=2)
    assert external.truncated and external.metrics['independence_fraction'] is None
    assert sum(t.truncated for t in external.transitions)==2
    with pytest.raises(ValueError):
        Learner().update([external])
    finite=collect(CFG,Fixed(0),1)
    assert not finite.truncated and finite.metrics['horizon_complete']


def test_late_birth_exclusion_and_maturation_success_after_death():
    cfg=CFG.variant('births',n_food=0,n_initial_agents=1,energy_start=60,
                    energy_cost_per_tick=0,food_respawn_rate=0,n_ticks=4,reproduce_threshold=30)
    ep=collect(cfg,Fixed(5),1)
    assert ep.metrics['births']==2 and ep.metrics['eligible_births']==2
    assert ep.metrics['independence_fraction']==1
    short=collect(cfg.variant('short',n_ticks=2),Fixed(5),1)
    assert short.metrics['eligible_births']==0
    assert short.metrics['excluded_late_births']==2
    assert short.metrics['independence_fraction'] is None


def test_weighted_metric_denominators_and_exposure():
    records=[]
    for initial in (family(),family(far=True)):
        records.append(collect(CFG.variant('exposure',n_ticks=3),Fixed(0),1,initial=initial).metrics)
    result=summarize(records)
    assert result['attendance_fraction']==.5
    assert result['care_ticks']==2 and result['adult_decisions']==10
    for r in records:
        assert r['starting_population']+r['births']-r['deaths']==r['final_population']
        assert r['adult_decisions']>=r['surviving_adult_ticks']
        assert sum(r['deaths_by_cause'].values())==r['deaths']


def test_seeds_and_frozen_matched_worlds():
    policies=[Policy('linear',1),Policy('deep',2),Uniform(),HandDesigned()]
    hashes=[]
    for p in policies:
        p.train()
        before=state_hash(p)
        records=evaluate(CFG,p,BLOCKS)
        assert p.training and before==state_hash(p)
        hashes.append([r['initial_hash'] for r in records])
    assert all(h==hashes[0] for h in hashes)
    torch.rand(20)
    assert evaluate(CFG,policies[0],BLOCKS)==evaluate(CFG,policies[0],BLOCKS)
    with pytest.raises(ValueError):
        evaluate(CFG,policies[0],BLOCKS,split='final_test')


@pytest.mark.parametrize('blocks', [dict(validation=(1000,)),dict(validation=(200001,200001)),
                                    dict(validation=(300001,)),dict(final_test=(-1,))])
def test_seed_partition_rejects_overlap(blocks):
    with pytest.raises(ValueError):
        replace(BLOCKS,**blocks)


def test_initialization_and_collection_preserve_ambient_rngs():
    torch_before=torch.get_rng_state().clone()
    py_before=random.getstate()
    np_before=np.random.get_state()
    p=Policy('deep',3)
    first=collect(CFG,p,4)
    second=collect(CFG,p,4)
    assert first==second
    assert torch.equal(torch_before,torch.get_rng_state())
    assert py_before==random.getstate()
    np_after=np.random.get_state()
    assert np.array_equal(np_before[1],np_after[1])
    assert first.transitions!=collect(CFG,p,5).transitions


@pytest.mark.parametrize('algorithm',('reinforce','reinforce_value'))
def test_checkpoint_optimizer_resume_and_identical_replay(tmp_path,algorithm):
    exp=Experiment(CFG,LearnerConfig(algorithm=algorithm,seed=1),seeds=BLOCKS)
    exp.step()
    checkpoint=tmp_path/'state.pt'
    exp.checkpoint(checkpoint)
    restored=Experiment.restore(checkpoint)
    assert restored.training_decisions==exp.training_decisions
    assert state_hash(restored.learner.policy)==state_hash(exp.learner.policy)
    assert collect(CFG,restored.learner.policy,22)==collect(CFG,exp.learner.policy,22)
    a,b=exp.step(),restored.step()
    a.pop('training_seconds');b.pop('training_seconds')
    assert a==b
    assert state_hash(restored.learner.policy)==state_hash(exp.learner.policy)
    with pytest.raises(FileExistsError):
        exp.checkpoint(checkpoint)


@pytest.mark.parametrize('algorithm',('reinforce','reinforce_value'))
def test_validation_frequency_does_not_change_training(algorithm):
    experiments=[Experiment(CFG,LearnerConfig(algorithm=algorithm,seed=2),seeds=BLOCKS,
                             validation_every=f) for f in (0,1,2)]
    for exp in experiments:
        exp.step();exp.step()
    assert len({exp.training_decisions for exp in experiments})==1
    assert len({state_hash(exp.learner.policy) for exp in experiments})==1
    assert experiments[0].validation_decisions==0 < experiments[1].validation_decisions
    independent=Experiment(CFG,LearnerConfig(algorithm=algorithm,seed=3),seeds=BLOCKS)
    independent.step();independent.step()
    assert state_hash(independent.learner.policy)!=state_hash(experiments[0].learner.policy)


def test_legacy_model_transfer_and_original_collector_metrics(tmp_path):
    legacy=PolicyNet(seed=1)
    file=tmp_path/'legacy.pt'
    torch.save(legacy.state_dict(),file)
    modern=load_legacy(file)
    x=torch.tensor([[.1]*12])
    assert torch.equal(legacy(x),modern(x))
    # Deterministic stays align action sequences despite different sampling engines.
    with torch.no_grad():
        for p in legacy.parameters():
            p.zero_()
        legacy.out.bias[0]=1000
    torch.save(legacy.state_dict(),file)
    modern=load_legacy(file)
    logs,rewards,stats=run_episode(CFG,legacy,False,5)
    ep=collect(CFG,modern,5)
    assert len(ep.transitions)==sum(len(v) for v in logs.values())
    assert ep.metrics['births']==stats['births']
    assert ep.metrics['final_population']==stats['final_population']
    assert sum(e.amount for e in ep.events)==pytest.approx(sum(map(sum,rewards.values())))


def test_hand_designed_observation_only_actions():
    x=torch.zeros(4,12)
    x[0,9]=1;x[0,11]=.9;x[0,3]=.6
    x[1,0]=-1;x[1,2]=.9;x[1,3]=.2
    x[2,3]=.8;x[2,8]=1
    assert HandDesigned()(x).argmax(-1)[:3].tolist()==[3,4,5]
    assert torch.equal(HandDesigned()(x)[3],torch.zeros(6))


def test_budget_counts_and_guard():
    exp=Experiment(CFG,seeds=BLOCKS,validation_every=0)
    result=exp.train_to_budget(13,max_episodes=3)
    assert result['realized']>=13
    assert result['overshoot'] < CFG.n_ticks*CFG.max_population
    assert exp.validation_decisions==0
    with pytest.raises(RuntimeError):
        Experiment(CFG,seeds=BLOCKS,validation_every=0).train_to_budget(100000,max_episodes=1)


@pytest.mark.parametrize('algorithm',('reinforce','reinforce_value'))
def test_synthetic_solvable_bandit_learns(algorithm):
    learner=Learner(LearnerConfig(architecture='linear',algorithm=algorithm,actor_lr=.03))
    # Rewards are action 3 = 1, every other action = 0. Each sample is a one-step world.
    x=torch.ones(1,12)
    initial=float(learner.policy.probabilities(x)[0,3].detach())
    losses=[]
    for update in range(40):
        with torch.no_grad():
            probs=learner.policy.probabilities(x)[0]
        # Draw a genuine on-policy batch with a local generator.
        actions=torch.multinomial(probs,64,replacement=True,
                                  generator=torch.Generator().manual_seed(update))
        episodes=[]
        for action in actions.tolist():
            episodes.append(Episode(update,update,'bandit',1,
                [Transition(0,0,tuple(x[0].tolist()),action,float(probs[action].log()))],
                [Event(0,0,float(action==3),'bandit')]))
        losses.append(learner.update(episodes)['actor_loss'])
    final=float(learner.policy.probabilities(x)[0,3].detach())
    assert final > .95 and final > initial+.5
    assert all(np.isfinite(losses))


def test_stale_rollout_is_rejected():
    learner=Learner()
    ep=collect(CFG,learner.policy,1)
    learner.update([ep])
    with pytest.raises(ValueError,match='consumed'):
        learner.update([ep])


def test_smoke_outputs_plot_and_overwrite_guard(tmp_path):
    out=tmp_path/'smoke'
    exp=run_smoke(out,architecture='linear',algorithm='reinforce_value')
    assert exp.learner.updates==2
    lines=[json.loads(line) for line in (out/'training.jsonl').read_text().splitlines()]
    for row in lines:
        checkpoint=out/row['checkpoint_id']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['checkpoint_sha256']
        assert row['validation']['worlds']
    assert not json.loads((out/'manifest.json').read_text())['final_test_executed']
    plot=learning_curves(out/'training.jsonl',out/'curve.png')
    assert plot.stat().st_size>1000
    with pytest.raises(FileExistsError):
        run_smoke(out)
    with pytest.raises(ValueError):
        run_smoke(ROOT/'results'/'forbidden-v2')


def test_zero_decisions_does_not_update():
    learner=Learner()
    ep=collect(CFG.variant('empty',n_initial_agents=0),learner.policy,1)
    assert learner.update([ep])['optimizer_updates']==0
    assert ep.metrics['attendance_fraction'] is None


@pytest.mark.parametrize('bad', [dict(algorithm='ppo'),dict(gamma=1.1),dict(actor_lr=0),
                                dict(reward_scope='unknown'),dict(max_grad_norm=-1)])
def test_configuration_rejects_invalid_options(bad):
    with pytest.raises(ValueError):
        LearnerConfig(**bad)


def test_legacy_posthumous_credit_is_compressed_to_last_action(monkeypatch):
    import rl
    import agent
    monkeypatch.setattr(rl,'draw_lifespan',lambda cfg,rng:1)
    monkeypatch.setattr('v2.environment.draw_lifespan',lambda cfg,rng:1)
    monkeypatch.setattr(agent,'draw_lifespan',lambda cfg,rng:100)
    cfg=CFG.variant('legacy-credit',n_initial_agents=1,n_food=0,food_respawn_rate=0,
                    n_ticks=4,dependency_ticks=3,neglect_tolerance=10,
                    energy_start=60,energy_cost_per_tick=0,ageing_enabled=True,
                    maturity_age=1,fertility_end_age=1,max_age=100)
    class BirthThenStay(torch.nn.Module):
        def __init__(self):
            super().__init__();self.calls=0
        def forward(self,x):
            self.calls+=1
            return Fixed(5 if self.calls==1 else 0)(x)
    _,rewards,_=rl.run_episode(cfg,BirthThenStay(),True,0)
    parent=next(iter(rewards))
    assert not parent.alive and rewards[parent]==[1.]
    ep=collect(cfg,BirthThenStay(),0,RewardConfig('offspring'))
    event=next(e for e in ep.events if e.kind=='independence')
    assert event.tick==3 and event.recipient==0
    assert event_returns(ep,.5,'individual_surrogate')[0]==.125


def test_all_equal_legacy_return_normalization_erases_signal():
    logits=torch.zeros(2,requires_grad=True)
    returns=torch.ones(2)
    centered=(returns-returns.mean())/(returns.std(unbiased=False)+1e-8)
    logp=logits.log_softmax(0)[1].expand(2)
    loss=-(logp*centered).mean()
    loss.backward()
    assert torch.equal(logits.grad,torch.zeros(2))
    raw_logits=torch.zeros(2,requires_grad=True)
    (-(raw_logits.log_softmax(0)[1].expand(2)*returns).mean()).backward()
    assert raw_logits.grad[1] < 0
    # This finite-batch effect is a methodological choice, not a NaN bug.


def test_world_objective_includes_cross_agent_causal_terms():
    ep=Episode(1,1,'x',2,[trajectory(0,0),trajectory(0,1)],
               [Event(1,1,1.,'offspring',2)])
    assert event_returns(ep,1.,'colony')==[1.,1.]
    assert event_returns(ep,1.,'individual_surrogate')==[0.,1.]


def test_baseline_weights_separate_and_updated():
    learner=Learner(LearnerConfig(algorithm='reinforce_value'))
    assert not {id(p) for p in learner.policy.parameters()} & {id(p) for p in learner.value.parameters()}
    before=state_hash(learner.value)
    result=learner.update([collect(CFG,learner.policy,1)])
    assert result['value_loss'] is not None and state_hash(learner.value)!=before
    assert np.isfinite(result['value_gradient_norm'])


def test_population_cap_does_not_create_phantom_births():
    cfg=CFG.variant('cap',n_initial_agents=1,max_population=1,n_food=0,
                    energy_start=60,energy_cost_per_tick=0)
    ep=collect(cfg,Fixed(5),0,RewardConfig('offspring'))
    assert ep.metrics['births']==0 and ep.metrics['final_population']==1
    assert ep.metrics['independent_offspring']==0


def test_maturation_tick_death_is_not_independence():
    def initial(cfg,seed):
        world,rng=family(pup_ages=(1,))(cfg,seed)
        world.agents[1].energy=0
        world.agents[1].lifespan=2
        return world,rng
    ep=collect(CFG.variant('die-maturing',n_food=0,food_respawn_rate=0),Fixed(0),0,
               RewardConfig('offspring'),initial=initial)
    assert ep.metrics['independent_offspring']==0
    assert not any(e.kind=='independence' for e in ep.events)


def test_invalid_rewards_and_nonfinite_policy_rejected():
    with pytest.raises(ValueError):
        RewardConfig('not-an-objective')
    with pytest.raises(ValueError):
        RewardConfig(survival=float('nan'))
    class Bad(torch.nn.Module):
        def forward(self,x):
            return torch.full((len(x),6),float('nan'))
    with pytest.raises(ValueError):
        collect(CFG,Bad(),0)


@pytest.mark.parametrize('seed',range(6))
def test_stochastic_environment_parity_with_legacy(seed):
    cfg=CFG.variant('parity',n_ticks=20,n_initial_agents=4,n_food=10,max_population=20,
                    dependency_ticks=3)
    observed=[]
    legacy=PolicyNet(seed=3)
    logs,rewards,stats=run_episode(cfg,legacy,True,seed,senses_sink=observed)
    episode=collect(cfg,Policy('original',seed=3),seed,RewardConfig('offspring'))
    assert np.array_equal(np.concatenate(observed),np.array([t.observation for t in episode.transitions]))
    assert stats['births']==episode.metrics['births']
    assert stats['final_population']==episode.metrics['final_population']
    assert sum(map(len,logs.values()))==episode.decisions
    assert sum(map(sum,rewards.values()))==pytest.approx(episode.metrics['total_reward'])


def test_checkpoint_source_drift_is_rejected(tmp_path):
    exp=Experiment(CFG,seeds=BLOCKS)
    original=tmp_path/'original.pt'
    exp.checkpoint(original)
    data=torch.load(original,weights_only=True)
    data['provenance']['source_sha256']['agent.py']='different-source'
    tampered=tmp_path/'drift.pt'
    torch.save(data,tampered)
    with pytest.raises(ValueError,match='source_sha256'):
        Experiment.restore(tampered)
