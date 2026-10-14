"""Resource-neutral crowding intervention and unchanged focal dynamics."""
import json
import random
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT)]
from brain import ACTIONS, SENSE_INDEX
from config import UNIVERSE_25
from rl import PolicyNet
from experiments import controlled_care as assay


def fixed_policy(action='stay'):
    policy = PolicyNet(seed=0)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.out.bias[ACTIONS.index(action)] = 100
    policy.eval()
    return policy


def place_parent_near_pup(monkeypatch, lifespan=None):
    original = assay.setup
    def setup(*args):
        result = original(*args)
        _,_,_,parent,pup,_,_ = result
        parent.x = pup.x+2 if pup.x+2 < UNIVERSE_25.width else pup.x-2
        parent.y = pup.y
        if lifespan is not None:
            parent.lifespan = lifespan
        return result
    monkeypatch.setattr(assay,'setup',setup)


@pytest.mark.parametrize('seed',[300000,300001,300002])
@pytest.mark.parametrize('energy',[10,20])
def test_matched_initial_conditions_resources_and_background_geometry(seed,energy):
    setups = [assay.setup(seed,energy,n,d) for n in (0,8) for d in (False,True)]
    assert len({assay.digest(s[-1]) for s in setups}) == 1
    assert len({repr(s[1].getstate()) for s in setups}) == 1
    for cfg,rng,world,parent,pup,background,initial in setups:
        assert not cfg.crowding_cost_enabled
        assert cfg.reproduce_threshold == float('inf')
        assert parent.age == cfg.maturity_age and not parent.is_dependent
        assert pup.parent is parent and parent.dependents == [pup]
        assert pup.age==0 and pup.dependent_until==60 and pup.neglect_ticks==0
        assert world.occupied_nests == {(pup.x,pup.y)}
        assert len({(b.x,b.y) for b in background})==len(background)
        assert all(max(abs(b.x-pup.x),abs(b.y-pup.y))<=2 for b in background)
        assert all((b.x,b.y) not in world.nests | world.food for b in background)
        assert all(not hasattr(b,'act') and not hasattr(b,'energy') for b in background)
        world.rebuild_occupancy(cfg.crowding_radius)
        assert world.count_neighbours(pup.x,pup.y,2,exclude=pup)==len(background)


def test_only_crowding_input_changes_and_original_training_config_is_unchanged():
    low = assay.setup(300000,20,0,False)
    high = assay.setup(300000,20,8,False)
    vectors = []
    for cfg,_,world,parent,*_ in (low,high):
        world.rebuild_occupancy(cfg.crowding_radius)
        vectors.append(np.array(parent.sense(world,cfg)))
    index = SENSE_INDEX['crowding']
    assert np.array_equal(np.delete(vectors[0],index),np.delete(vectors[1],index))
    assert UNIVERSE_25.crowding_cost_enabled
    assert not UNIVERSE_25.crowding_blocks_care
    assert UNIVERSE_25.reproduce_threshold==35


def test_background_is_immutable():
    with pytest.raises(FrozenInstanceError):
        assay.BackgroundBody(1,2).x=3


def test_metabolism_background_resources_and_full_parent_followup(monkeypatch):
    place_parent_near_pup(monkeypatch)
    results = [assay.trial(fixed_policy(),300000,20,n,d)
               for n in (0,8) for d in (False,True)]
    for r in results:
        assert r['elapsed_ticks']==len(r['history'])==60
        assert r['counts']['parent_meals']==0
        assert r['history'][-1]['parent_energy_after']==pytest.approx(5)
        assert all(t['food_count']==60 for t in r['history'])
        assert len({t['food_hash'] for t in r['history']})==1
        assert r['parent_alive'] and r['births']==0
        assert r['starting_population']-r['deaths']==r['final_population']
        assert r['final_total_occupancy_bodies']==r['final_population']+r['background_population']
    assert [r['independent'] for r in results]==[True,True,True,False]
    assert results[-1]['deaths_by_cause']['neglect']==1
    assert results[-1]['counts']['crowd_blocked_ticks']==26
    assert all(r['counts']['crowd_blocked_ticks']==0 for r in results[:-1])
    # Density does not alter actions, feeding or metabolic costs for a crowd-insensitive policy.
    assert all([t['actions'].get('parent') for t in r['history']]==['stay']*60 for r in results)


def test_starvation_counted_once_with_full_horizon_and_orphan_independence(monkeypatch):
    place_parent_near_pup(monkeypatch)
    r = assay.trial(fixed_policy(),300000,10,8,False)
    assert not r['parent_alive'] and r['parent_cause_of_death']=='starvation'
    assert r['deaths_by_cause']=={'starvation':1,'old age':0,'neglect':0}
    assert r['elapsed_ticks']==60 and r['independent']
    assert r['final_population']==1


def test_old_age_and_neglect_deaths_are_separate_and_counted_once(monkeypatch):
    place_parent_near_pup(monkeypatch,lifespan=130)
    r = assay.trial(fixed_policy(),300000,20,0,False)
    assert r['deaths_by_cause']=={'starvation':0,'old age':1,'neglect':1}
    assert r['parent_cause_of_death']=='old age' and r['pup_cause_of_death']=='neglect'
    assert r['deaths']==2 and r['final_population']==0 and r['elapsed_ticks']==60


def test_reproduction_actions_do_not_create_additional_pups():
    r = assay.trial(fixed_policy('reproduce'),300000,20,8,False)
    assert r['births']==0
    assert r['history'][-1]['parent_energy_after']==pytest.approx(5)
    assert r['counts']['parent_action_ticks']==60


def test_food_consumption_and_resource_rng_match_across_background_conditions(monkeypatch):
    original=assay.setup
    def setup(*args):
        result=original(*args)
        _,_,world,parent,*_=result
        parent.x,parent.y=sorted(world.food)[0]
        return result
    monkeypatch.setattr(assay,'setup',setup)
    low=assay.trial(fixed_policy(),300000,20,0,False)
    high=assay.trial(fixed_policy(),300000,20,8,False)
    assert low['counts']['parent_meals']==high['counts']['parent_meals']>=1
    assert low['parent_alive']==high['parent_alive']
    for a,b in zip(low['history'],high['history']):
        for key in ('food_hash','parent_position','parent_energy_after','parent_alive','actions'):
            assert a[key]==b[key]
    assert low['history'][0]['parent_energy_after']==pytest.approx(20-.25+12)


def test_different_world_seeds_change_initial_geometry():
    assert assay.digest(assay.setup(300000,20,8,True)[-1]) != assay.digest(
        assay.setup(300001,20,8,True)[-1])


def test_seeded_replay_frozen_weights_and_global_rng_restoration():
    policy = PolicyNet(seed=0)
    before = assay.policy_hash(policy)
    state = torch.random.get_rng_state().clone()
    a = assay.trial(policy,300001,10,8,True)
    assert torch.equal(state,torch.random.get_rng_state())
    random.seed(123); random.random(); np.random.seed(456); np.random.rand(); torch.rand(12)
    assert a==assay.trial(policy,300001,10,8,True)
    assert assay.policy_hash(policy)==before
    assert all(p.grad is None for p in policy.parameters())


def test_zero_background_preserves_existing_parent_snapshot_timing():
    policy = PolicyNet(seed=1)
    off = assay.trial(policy,300001,20,0,False)
    on = assay.trial(policy,300001,20,0,True)
    assert on['counts']['third_party_crowd_sum']==0
    assert on['counts']['crowd_blocked_ticks']>0
    event=next(t for t in on['history'] if t['care'] and t['care']['crowd_blocked'])
    assert event['care']['crowd']==1 and event['care']['baseline_attendance']
    assert event['care']['effective_radius']==pytest.approx(3/1.1)
    assert off['counts']['crowd_blocked_ticks']==0


def test_impossible_matched_geometry_fails_clearly(monkeypatch):
    original = assay._build_world
    def build(cfg,rng):
        world=original(cfg,rng); world.nests={(0,0)}; world.food={(0,0)}
        return world
    monkeypatch.setattr(assay,'_build_world',build)
    with pytest.raises(ValueError,match='No matched care geometry'):
        assay.setup(300000,20,8,True)


def test_existing_output_and_legacy_paths_are_refused(tmp_path):
    target=tmp_path/'existing'; target.mkdir(); (target/'marker').write_text('preserve')
    args=SimpleNamespace(out=str(target),seeds='0',world_seeds='300000')
    with pytest.raises(FileExistsError):
        assay.execute(args)
    assert (target/'marker').read_text()=='preserve'
    args.out=str(ROOT/'results/final_reward_v1/new')
    with pytest.raises(ValueError,match='outside existing final results'):
        assay.execute(args)


def test_summary_preserves_mortality_care_counts_and_training_pair_units():
    records=[]
    for objective in ('survival','offspring'):
        for energy in (10,20):
            for n in (0,8):
                for disruption in (False,True):
                    r=assay.trial(fixed_policy(),300000,energy,n,disruption)
                    r.update(objective=objective,training_seed=0); records.append(r)
    summary=assay.summarize(records)
    assert len(summary['groups'])==16 and len(summary['training_pair_rates'])==4
    assert all(p['difference']==0 for p in summary['training_pair_rates'])
    assert sum(g['deaths_by_cause']['starvation'] for g in summary['groups'])==sum(
        r['deaths_by_cause']['starvation'] for r in records)
    assert sum(g['counts']['care_ticks'] for g in summary['groups'])==sum(
        r['counts']['care_ticks'] for r in records)
