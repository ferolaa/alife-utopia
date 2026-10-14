"""Checks for observation-only outcome metrics and isolated experiment output."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from config import DEFAULT
from rl import PolicyNet, run_episode
from experiments.final_rewards import natural_episode, assigned_pup_trial, state_hash, execute


def reproducer():
    policy=PolicyNet(seed=0)
    with torch.no_grad():
        for p in policy.parameters():
            p.zero_()
        policy.out.bias[5]=100
    return policy


def config(**changes):
    return DEFAULT.variant('metrics',n_initial_agents=1,n_food=0,
                           energy_start=60,energy_cost_per_tick=0,
                           parental_care_enabled=True,dependency_ticks=2,
                           reproduce_threshold=30,**changes)


def test_natural_observer_preserves_episode_and_counts_actual_independence():
    cfg=config(n_ticks=4)
    policy=reproducer()
    _,_,expected=run_episode(cfg,policy,False,seed=2)
    before=state_hash(policy)
    actual=natural_episode(cfg,policy,2)
    assert all(actual[k]==v for k,v in expected.items())
    assert actual['births']==actual['eligible_births']==actual['independent_offspring']==2
    assert actual['independence_fraction']==1
    assert state_hash(policy)==before


def test_late_births_are_excluded_and_no_births_are_not_zero_success():
    actual=natural_episode(config(n_ticks=2),reproducer(),2)
    assert actual['eligible_births']==0 and actual['excluded_late_births']==2
    assert actual['independence_fraction'] is None
    actual=natural_episode(config(n_ticks=4,max_population=1),reproducer(),2)
    assert actual['births']==0 and actual['independence_fraction'] is None


def test_pup_dying_on_maturation_tick_is_not_success():
    actual=natural_episode(config(n_ticks=4,reproduce_cost=0),reproducer(),2)
    assert actual['eligible_births']>0 and actual['independent_offspring']==0


def test_assigned_pup_trials_replay_and_do_not_update_weights():
    policy=PolicyNet(seed=0)
    before=state_hash(policy)
    first=assigned_pup_trial(policy,100000,20)
    torch.rand(20)
    assert first==assigned_pup_trial(policy,100000,20)
    assert state_hash(policy)==before
    assert first['care_ticks']<=59


def test_existing_output_directory_is_refused(tmp_path):
    target=tmp_path/'existing'
    target.mkdir()
    marker=target/'marker';marker.write_text('preserve')
    with pytest.raises(FileExistsError):
        execute(SimpleNamespace(out=str(target)))
    assert marker.read_text()=='preserve'


def test_paired_analysis_uses_training_pairs_and_omits_single_pair_inference():
    from experiments.final_rewards import paired_analysis
    result=paired_analysis([{'difference':.1},{'difference':.2}])
    assert result['n_training_pairs']==2
    assert result['mean_difference']==pytest.approx(.15)
    assert result['exact_sign_flip_p']==.5
    assert result['paired_bootstrap_95_percent_interval']==pytest.approx([.1,.2])
    assert paired_analysis([{'difference':.1}])['exact_sign_flip_p'] is None


def assert_accounting(record):
    assert record['starting_population'] + record['births'] - record['deaths'] == record['final_population']
    if 'deaths_by_cause' in record:
        assert sum(record['deaths_by_cause'].values()) == record['deaths']
        for cause in record['deaths_by_cause']:
            assert record['founder_deaths_by_cause'][cause] + record['offspring_deaths_by_cause'][cause] == record['deaths_by_cause'][cause]


def test_founder_starvation_is_counted_once():
    cfg=DEFAULT.variant('starvation', n_ticks=5,n_initial_agents=2,n_food=0,
                        energy_start=.1,energy_cost_per_tick=1)
    record=natural_episode(cfg,PolicyNet(seed=0),1)
    assert record['deaths_by_cause']=={'starvation':2,'old age':0,'neglect':0}
    assert record['founder_deaths_by_cause']['starvation']==2
    assert record['elapsed_ticks']==1
    assert_accounting(record)


def test_founder_and_offspring_old_age_are_counted_once():
    cfg=config(n_ticks=5,ageing_enabled=True,max_age=2,max_age_spread=0,
               maturity_age=1,fertility_end_age=1).variant('age',dependency_ticks=1)
    record=natural_episode(cfg,reproducer(),2)
    assert record['founder_deaths_by_cause']['old age']==1
    assert record['offspring_deaths_by_cause']['old age']==1
    assert record['deaths']==2 and record['independent_offspring']==1
    assert_accounting(record)


def test_offspring_starvation_is_counted_once():
    record=natural_episode(config(n_ticks=4,reproduce_cost=0),reproducer(),2)
    assert record['offspring_deaths_by_cause']['starvation']==2
    assert record['founder_deaths_by_cause']['starvation']==0
    assert_accounting(record)


def test_offspring_neglect_is_counted_once(monkeypatch):
    import rl
    from world import World
    import random
    w=World(10,10,0,wrap_edges=False,rng=random.Random(0))
    w.nests={(5,5)}
    monkeypatch.setattr(w,'random_square',lambda:(5,5))
    monkeypatch.setattr(rl,'_build_world',lambda cfg,rng:w)
    calls=[]
    def choices(dist,sample_shape=torch.Size()):
        action=5 if not calls else 3
        calls.append(True)
        return torch.tensor([action])
    monkeypatch.setattr(torch.distributions.Categorical,'sample',choices)
    record=natural_episode(config(n_ticks=5,nests_enabled=True,care_radius=0,neglect_tolerance=0),reproducer(),2)
    assert record['deaths_by_cause']=={'starvation':0,'old age':0,'neglect':1}
    assert record['offspring_deaths_by_cause']['neglect']==1
    assert_accounting(record)


def test_assigned_parent_starvation_records_cause_and_terminal_status(monkeypatch):
    import experiments.final_rewards as runner
    monkeypatch.setattr(runner,'UNIVERSE_25',runner.UNIVERSE_25.variant('known_death',energy_cost_per_tick=1000))
    record=runner.assigned_pup_trial(PolicyNet(seed=0),100000,10)
    assert record['parent_outcome']=='dead'
    assert record['parent_cause_of_death']=='starvation'
    assert not record['parent_alive']
    assert_accounting(record)


@pytest.mark.parametrize('return_to_pup', [False,True])
def test_assigned_parent_survival_distinguishes_early_stop_from_full_horizon(monkeypatch,return_to_pup):
    baseline=assigned_pup_trial(PolicyNet(seed=0),100000,20)
    initial=baseline['initial_conditions']
    dx=initial['nest'][0]-initial['parent_position'][0]
    dy=initial['nest'][1]-initial['parent_position'][1]
    toward=3 if dx>0 else 4 if dx<0 else 2 if dy>0 else 1
    calls=[]
    def choices(dist,sample_shape=torch.Size()):
        action=toward if return_to_pup and not calls else 0
        calls.append(True)
        return torch.tensor([action]+[0]*(dist.logits.shape[0]-1))
    monkeypatch.setattr(torch.distributions.Categorical,'sample',choices)
    record=assigned_pup_trial(PolicyNet(seed=0),100000,20)
    assert record['parent_alive'] and record['parent_cause_of_death'] is None
    assert record['parent_outcome']==('survived_full_trial' if return_to_pup else 'alive_at_early_stop')
    assert record['elapsed_ticks']==(60 if return_to_pup else 26)
    assert record['independent']==return_to_pup
    assert_accounting(record)


def test_mortality_totals_reach_saved_summary():
    from experiments.final_rewards import outcome_summary
    cfg=DEFAULT.variant('starvation',n_ticks=2,n_initial_agents=1,n_food=0,
                        energy_start=.1,energy_cost_per_tick=1)
    natural=natural_episode(cfg,PolicyNet(seed=0),1)
    trial=assigned_pup_trial(PolicyNet(seed=0),100000,10)
    summary=outcome_summary([{'objective':mode,'natural_episodes':[natural],
                             'assigned_pup_trials':[trial]} for mode in ('survival','offspring')])
    for mode in ('survival','offspring'):
        assert summary[mode]['deaths_by_cause']['starvation']==1
        assert summary[mode]['founder_deaths_by_cause']['starvation']==1
        assert sum(summary[mode]['assigned_parent_outcomes'].values())==1
