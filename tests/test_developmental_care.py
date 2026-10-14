"""Regression coverage for the separate approved Phase 3 feasibility assay."""
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from brain import ACTIONS
from config import UNIVERSE_25
from rl import PolicyNet
from experiments import developmental_care as assay


def fixed_policy(action='stay'):
    p=PolicyNet(seed=0)
    with torch.no_grad():
        for parameter in p.parameters():parameter.zero_()
        p.out.bias[ACTIONS.index(action)]=100
    p.eval();return p


@pytest.mark.parametrize('disrupted,damage,expected',[(False,False,0),(False,True,0),(True,False,0),(True,True,.3)])
def test_childhood_dose_maturation_survival_and_first_rl_action(disrupted,damage,expected):
    subject,r=assay.childhood(fixed_policy('east'),410000,disrupted,damage)
    assert r['counts']['care_ticks']==59
    assert r['counts']['unattended_ticks']==(18 if disrupted else 0)
    unattended=[h['subject']['age'] for h in r['history'] if h['attended'] is False]
    assert unattended==(list(range(10,28)) if disrupted else [])
    assert max(h['subject']['neglect_ticks'] for h in r['history'])==(18 if disrupted else 0)
    assert r['before_maturation']['age']==59 and r['before_maturation']['neglect_ticks']==0
    assert r['before_maturation']['impairment']==0
    assert subject.age==60 and subject.impairment==expected and subject.parent is None
    assert r['history'][-1]['action']=='east' and r['history'][-1]['occupied_nests']==0
    assert all(h['action'] is None for h in r['history'][:-1])
    assert subject.energy==17.75 and subject.alive
    assert r['caregiver_final']['age']==180 and r['caregiver_final']['energy']==45
    assert r['caregiver_final']['alive'] and r['counts']['caregiver_meals']==0
    assert r['deaths_by_cause']=={'starvation':0,'old age':0,'neglect':0}
    assert all(h['crowd']==9 for h in r['history'][:-1])


@pytest.mark.parametrize('index',[0,1,2])
def test_matched_childhood_physics_and_resource_neutral_background(index):
    setups=[assay.childhood_setup(410000+index,d) for d in (False,True)]
    for cfg,rng,world,giver,subject,bg in setups:
        assert len(bg)==8 and len({(b.x,b.y) for b in bg})==8
        assert max(abs(subject.x-giver.x),abs(subject.y-giver.y))==2
        assert all(max(abs(subject.x-b.x),abs(subject.y-b.y))<=2 for b in bg)
        assert all(not hasattr(b,'act') and not hasattr(b,'energy') for b in bg)
        assert all((b.x,b.y) not in world.food|world.nests for b in bg)
        assert not cfg.crowding_cost_enabled and cfg.reproduce_threshold==float('inf')
    assert assay.geometry(setups[0][2])==assay.geometry(setups[1][2])
    assert setups[0][1].getstate()==setups[1][1].getstate()
    p=fixed_policy();_,normal=assay.childhood(p,410000+index,False,True)
    _,disrupted=assay.childhood(p,410000+index,True,True)
    assert normal['initial']==disrupted['initial']
    for n,d in zip(normal['history'],disrupted['history']):
        assert n['caregiver']==d['caregiver']
        assert n['food_hash']==d['food_hash']
        for key in ('position','age','energy','lifespan','alive','parent_attached','dependents'):
            assert n['subject'][key]==d['subject'][key]


def test_bridge_has_no_attrition_preserves_trait_and_matches_all_branches():
    p=PolicyNet(seed=0)
    records=[assay.trajectory(p,0,c,d) for c in (False,True) for d in (False,True)]
    for r in records:
        assert r['childhood_survivors']==r['mature_survivors']==r['adults_evaluated']==1
        bridge=r['bridge'];assert [h['age'] for h in bridge['history']]==list(range(61,121))
        assert bridge['final']['energy']>=2.75 and bridge['final']['age']==120
        assert all(h['alive'] for h in bridge['history'])
        assert bridge['final']['impairment']==r['childhood']['final']['impairment']==r['adult']['impairment']
    assert all(r['bridge']['history']==records[0]['bridge']['history'] for r in records)
    assert records[0]['adult']==records[1]['adult']==records[2]['adult']


def test_only_pup_inputs_change_at_same_state_and_initial_physics_match():
    p=PolicyNet(seed=1);normal=assay.trajectory(p,1,False,True);disrupted=assay.trajectory(p,1,True,True)
    assert normal['adult']['initial_hash']==disrupted['adult']['initial_hash']
    assert normal['adult']['initial']==disrupted['adult']['initial']
    n=np.array(normal['adult']['history'][0]['parent_policy_input'])
    d=np.array(disrupted['adult']['history'][0]['parent_policy_input'])
    assert np.array_equal(n[list(assay.OTHER_INPUTS)],d[list(assay.OTHER_INPUTS)])
    assert np.allclose(d[list(assay.PUP_INPUTS)],n[list(assay.PUP_INPUTS)]*.7)
    assert np.any(n[list(assay.PUP_INPUTS)]!=0)
    for h in disrupted['adult']['history']:
        actual=np.array(h['parent_policy_input']);reference=np.array(h['unattenuated_same_state_input'])
        assert np.array_equal(actual[list(assay.OTHER_INPUTS)],reference[list(assay.OTHER_INPUTS)])
        assert np.allclose(actual[list(assay.PUP_INPUTS)],reference[list(assay.PUP_INPUTS)]*.7)
    assert disrupted['adult']['final_impairment']==.3


def test_isolated_stage_streams_replay_and_frozen_weights(monkeypatch):
    p=PolicyNet(seed=1);before=assay.policy_hash(p);state=torch.random.get_rng_state().clone()
    expected=assay.trajectory(p,2,True,True)
    assert torch.equal(state,torch.random.get_rng_state())
    random.seed(9);random.random();np.random.seed(8);np.random.rand();torch.rand(9)
    original=assay._build_world
    def noisy(cfg,rng):
        random.random();np.random.rand(30);torch.rand(20)
        return original(cfg,rng)
    monkeypatch.setattr(assay,'_build_world',noisy)
    # Adult setup comes from the preserved runner and remains independently seeded.
    assert assay.trajectory(p,2,True,True)==expected
    assert assay.policy_hash(p)==before and all(parameter.grad is None for parameter in p.parameters())


def test_childhood_rng_activity_cannot_change_later_stages(monkeypatch):
    p=fixed_policy();expected=assay.trajectory(p,0,True,False)
    original=assay.childhood
    def noisy(*args):
        result=original(*args);torch.rand(17);random.random();np.random.rand(23);return result
    monkeypatch.setattr(assay,'childhood',noisy)
    actual=assay.trajectory(p,0,True,False)
    assert actual['bridge']==expected['bridge'] and actual['adult']==expected['adult']


def test_adult_metabolism_and_full_followup_after_early_pup_death():
    r=assay.trajectory(fixed_policy(),0,True,True)['adult']
    assert not r['independent'] and r['pup_cause_of_death']=='neglect'
    assert r['elapsed_ticks']==len(r['history'])==60
    assert r['parent_alive'] and r['history'][-1]['parent_age']==180
    assert r['history'][-1]['parent_energy']>=5
    assert r['starting_population']-r['deaths']==r['final_population']
    assert r['counts']['sensory_audit_ticks']==60


def test_assigned_pup_first_adult_action_uses_rl_and_never_placeholder_brain(monkeypatch):
    original=assay.adult_setup
    def setup(subject,seed):
        cfg,rng,world,pup,initial=original(subject,seed)
        subject.x=pup.x+2 if pup.x+2<cfg.width else pup.x-2
        subject.y=pup.y;initial['parent']['position']=[subject.x,subject.y]
        return cfg,rng,world,pup,initial
    def unexpected(*args,**kwargs):
        raise AssertionError('Placeholder brain used instead of frozen RL inference')
    monkeypatch.setattr(assay,'adult_setup',setup)
    monkeypatch.setattr(assay.Brain,'decide',unexpected)
    r=assay.trajectory(fixed_policy(),0,True,True)['adult']
    assert r['independent'] and r['history'][-1]['pup_age']==60
    assert r['history'][-1]['actions']['pup']=='stay'
    assert r['counts']['care_ticks']==59 and r['counts']['attended_ticks']==59


def test_ordinary_feeding_is_preserved(monkeypatch):
    original=assay.adult_setup
    def setup(subject,seed):
        cfg,rng,world,pup,initial=original(subject,seed)
        subject.x,subject.y=sorted(world.food)[0]
        initial['parent']['position']=[subject.x,subject.y]
        return cfg,rng,world,pup,initial
    monkeypatch.setattr(assay,'adult_setup',setup)
    r=assay.trajectory(fixed_policy(),0,True,True)['adult']
    assert r['counts']['parent_meals']>=1
    assert r['history'][0]['parent_energy']==pytest.approx(20-.25+12)
    assert all(h['food_count']==60 for h in r['history'])


def test_parent_starvation_and_pup_neglect_are_counted_once(monkeypatch):
    original=assay.adult_setup
    def setup(subject,seed):
        cfg,rng,world,pup,initial=original(subject,seed)
        subject.x,subject.y=pup.x,pup.y;subject.energy=.25
        return cfg,rng,world,pup,initial
    monkeypatch.setattr(assay,'adult_setup',setup)
    r=assay.trajectory(fixed_policy(),0,False,False)['adult']
    assert r['parent_cause_of_death']=='starvation' and not r['parent_alive']
    assert r['deaths_by_cause']=={'starvation':1,'old age':0,'neglect':1}
    assert r['deaths']==2 and r['final_population']==0 and len(r['history'])==60
    assert all(h['parent_age']==121 for h in r['history'])


def test_childhood_mortality_is_measured_and_missing_adult_is_not_imputed(monkeypatch):
    original=assay.childhood_setup
    def setup(seed,damage):
        result=original(seed,damage);result[3].lifespan=121;return result
    monkeypatch.setattr(assay,'childhood_setup',setup)
    r=assay.trajectory(fixed_policy(),0,False,False)
    assert r['childhood_survivors']==r['mature_survivors']==r['adults_evaluated']==0
    assert r['bridge'] is None and r['adult'] is None
    assert r['childhood']['deaths_by_cause']=={'starvation':0,'old age':1,'neglect':1}
    r.update(training_seed=0,objective='survival')
    summary=assay.summarize([r]);g=summary['groups'][0]
    assert g['assigned_cohort']==1 and g['adults_evaluated']==0
    assert g['independence_fraction'] is None and g['cohort_recruitment_fraction']==0
    assert summary['training_pair_effects'][0]['survival'] is None


def test_different_seed_triplets_change_geometry_and_original_configs_unchanged():
    a=assay.childhood_setup(410000,True);b=assay.childhood_setup(410001,True)
    assert assay.geometry(a[2])!=assay.geometry(b[2])
    assert UNIVERSE_25.crowding_cost_enabled and not UNIVERSE_25.developmental_damage_enabled
    with pytest.raises(ValueError,match='three approved'):
        assay.trajectory(fixed_policy(),3,True,True)


def test_output_protection_refuses_legacy_and_existing_directories(tmp_path):
    with pytest.raises(ValueError,match='separate new'):
        assay.execute(ROOT/'results/final_reward_v1/phase3')
    existing=tmp_path/'exists';existing.mkdir();(existing/'marker').write_text('preserve')
    with pytest.raises(FileExistsError):assay.execute(existing)
    assert (existing/'marker').read_text()=='preserve'


@pytest.mark.parametrize('full,expected',[(False,48),(True,1600)])
def test_matrix_count_metadata_cannot_be_overwritten_by_checkpoint_hash(monkeypatch,tmp_path,full,expected):
    def no_physics(policy,index,disrupted,damage,full=False):
        return {'world_index':index,'disrupted_childhood':disrupted,'damage_enabled':damage}
    monkeypatch.setattr(assay,'trajectory',no_physics)
    monkeypatch.setattr(assay,'summarize',lambda records:{'test_only_stub':True})
    monkeypatch.setattr(assay,'validate',lambda records,full=False:{'complete_unique_matrix':len(records)==expected})
    out=tmp_path/'matrix'
    summary=assay.execute(out,full=full)
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['expected_trajectories']==summary['expected_trajectories']==summary['trajectories']==expected
    assert type(manifest['expected_trajectories']) is int
    assert len(list(out.glob('*_s*.json')))==expected
