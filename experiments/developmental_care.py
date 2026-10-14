"""Phase 3 pilot/full evaluation: programmed cue attenuation, not neural damage.

The original Agent.act/care/growth rules are used unchanged. Childhood caregiver
actions are scripted stay, with eight occupancy-only bodies in every branch.
Only disrupted childhood enables care blocking at pup ages 10..27 inclusive.
Developmental assignment is OFF/ON; the stored trait is preserved through a real
age-61..120 uncrowded bridge, then a matched energy-20 assigned-pup assessment.
No learning, recovery, breeding or physiology of background bodies is introduced.
Default scope is the approved two-seed, three-world pilot; --full selects the
prespecified ten-seed, twenty-world matrix without altering the assay.
"""
import argparse
import copy
import json
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from agent import Agent,draw_lifespan
from brain import ACTIONS,Brain,SENSE_INDEX
from config import UNIVERSE_25
from rl import PolicyNet,_build_world
from experiments.controlled_care import BackgroundBody,digest,file_hash,policy_hash,setup as care_setup

PUP_INPUTS=tuple(SENSE_INDEX[k] for k in ('pup_dx','pup_dy','pup_need'))
OTHER_INPUTS=tuple(i for i in range(12) if i not in PUP_INPUTS)


def config(damage=False,disruption=False):
    return UNIVERSE_25.variant('developmental_care_pilot',n_ticks=60,
        crowding_cost_enabled=False,crowding_blocks_care=disruption,
        nest_intrusion_enabled=False,developmental_damage_enabled=damage,
        reproduce_threshold=float('inf'))


def snapshot(a):
    return {'position':[a.x,a.y],'age':a.age,'energy':a.energy,
            'lifespan':a.lifespan,'alive':a.alive,'cause_of_death':a.cause_of_death,
            'dependent_until':a.dependent_until,'impairment':a.impairment,
            'neglect_suffered':a.neglect_suffered,'neglect_ticks':a.neglect_ticks,
            'parent_attached':a.parent is not None,'dependents':len(a.dependents)}


def geometry(world):
    return {'food':sorted(world.food),'feeders':sorted(world.feeders),
            'nests':sorted(world.nests),'claimed_nests':sorted(world.occupied_nests)}


def actions(policy,world,cfg,movers):
    if not movers:return [],[]
    senses=[a.sense(world,cfg) for a in movers]
    logits=policy(torch.tensor(np.array(senses),dtype=torch.float32))
    picks=torch.distributions.Categorical(logits=logits).sample().tolist()
    return [ACTIONS[p] for p in picks],senses


def act_and_count_food(a,world,cfg,rng,action=None):
    eaten=[0];original=world.take_food
    def take(x,y):
        result=original(x,y);eaten[0]+=result;return result
    world.take_food=take
    try:
        child=a.act(world,cfg,rng,action=action)
    finally:
        world.take_food=original
    assert child is None,'Additional reproduction is excluded from this assay'
    return eaten[0]


def cleanup(a,world):
    if not a.alive:
        if a.nest is not None:
            world.release_nest(*a.nest);a.nest=None
        if a.parent is not None:a._leave_the_parent()


def childhood_setup(seed,damage):
    cfg=config(damage)
    rng=random.Random(seed);world=_build_world(cfg,rng)
    candidates=[]
    for x,y in sorted(world.nests):
        distance=min(max(abs(x-fx),abs(y-fy)) for fx,fy in world.food)
        if not 4<=distance<=8:continue
        for dx,dy in ((2,0),(-2,0),(0,2),(0,-2)):
            gx,gy=x+dx,y+dy
            if not (0<=gx<cfg.width and 0<=gy<cfg.height):continue
            sites=[(sx,sy) for sx in range(max(0,x-2),min(cfg.width,x+3))
                   for sy in range(max(0,y-2),min(cfg.height,y+3))
                   if (sx,sy) not in world.nests|world.food|{(gx,gy)}]
            if len(sites)>=8:candidates.append((x,y,gx,gy,sites))
    if not candidates:raise ValueError(f'No prespecified childhood geometry fits seed {seed}')
    x,y,gx,gy,sites=rng.choice(candidates)
    giver=Agent(gx,gy,60,Brain.random(rng),lifespan=draw_lifespan(cfg,rng));giver.age=120
    subject=Agent(x,y,18,Brain.random(rng),lifespan=draw_lifespan(cfg,rng))
    subject.parent=giver;subject.dependent_until=60;subject.nest=(x,y)
    giver.dependents=[subject];assert world.claim_nest(x,y)
    background=[BackgroundBody(*p) for p in random.Random(seed+2718281828).sample(sites,8)]
    world.agents=[giver,subject,*background]
    return cfg,rng,world,giver,subject,background


def childhood(policy,seed,disrupted,damage):
    cfg,rng,world,giver,subject,background=childhood_setup(seed,damage)
    initial={'subject':snapshot(subject),'caregiver':snapshot(giver),'geometry':geometry(world),
             'background':[[b.x,b.y] for b in background]}
    history=[];counts=Counter();maturation_before=None
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.random.default_generator.manual_seed(seed)
        for tick in range(1,61):
            cfg=config(damage,disrupted and 10<=subject.age+1<=27)
            world.rebuild_occupancy(cfg.crowding_radius)
            # The only RL action in childhood is the first adult action at age 60.
            selected,senses=actions(policy,world,cfg,[subject]) if subject.alive and subject.age+1>=60 else ([],[])
            if giver.alive:counts['caregiver_meals']+=act_and_count_food(giver,world,cfg,rng,'stay')
            dependent=subject.alive and subject.age+1<60
            crowd=world.count_neighbours(subject.x,subject.y,2,exclude=subject) if dependent else None
            before=subject.neglect_suffered
            if subject.alive:
                if subject.age==59:maturation_before=snapshot(subject)
                counts['subject_meals']+=act_and_count_food(subject,world,cfg,rng,selected[0] if selected else None)
            attended=subject.neglect_suffered==before if dependent else None
            if dependent:
                counts['care_ticks']+=1;counts['attended_ticks']+=attended
                counts['unattended_ticks']+=not attended;counts['crowd_sum']+=crowd
            cleanup(subject,world);cleanup(giver,world)
            world.agents=[a for a in (giver,subject) if a.alive]+background
            world.respawn_step()
            history.append({'tick':tick,'subject':snapshot(subject),'caregiver':snapshot(giver),
                'care_disruption':cfg.crowding_blocks_care,'attended':attended,'crowd':crowd,
                'action':selected[0] if selected else None,'policy_input':list(senses[0]) if senses else None,
                'food_count':len(world.food),'food_hash':digest(sorted(world.food)),
                'occupied_nests':len(world.occupied_nests),'focal_population':int(giver.alive)+int(subject.alive)})
    mortality={c:sum(not a.alive and a.cause_of_death==c for a in (giver,subject))
               for c in ('starvation','old age','neglect')}
    assert 2-sum(mortality.values())==int(giver.alive)+int(subject.alive)
    return subject,{'seed':seed,'initial':initial,'counts':dict(counts),'history':history,
        'before_maturation':maturation_before,'final':snapshot(subject),'caregiver_final':snapshot(giver),
        'cohort_assigned':1,'cohort_survivors':int(subject.alive),'deaths_by_cause':mortality}


def bridge(policy,subject,seed):
    cfg=config();rng=random.Random(seed);world=_build_world(cfg,rng)
    subject.x,subject.y=world.random_square();world.agents=[subject]
    initial={'subject':snapshot(subject),'geometry':geometry(world)};history=[];meals=0
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.random.default_generator.manual_seed(seed)
        for tick in range(1,61):
            world.rebuild_occupancy(cfg.crowding_radius)
            picks,senses=actions(policy,world,cfg,[subject]) if subject.alive else ([],[])
            if subject.alive:meals+=act_and_count_food(subject,world,cfg,rng,picks[0])
            world.agents=[subject] if subject.alive else []
            world.respawn_step()
            history.append({'tick':tick,'age':subject.age,'position':[subject.x,subject.y],
                'energy':subject.energy,'alive':subject.alive,'action':picks[0] if picks else None,
                'policy_input':list(senses[0]) if senses else None,'food_count':len(world.food),
                'food_hash':digest(sorted(world.food))})
    return {'seed':seed,'initial':initial,'history':history,'final':snapshot(subject),
            'meals':meals,'cohort_survivors':int(subject.alive)}


def adult_setup(subject,seed):
    cfg,rng,world,template,pup,_,_=care_setup(seed,20,0,False)
    assert subject.age==120 and subject.alive and subject.parent is None
    # Normalize assessment resources/position, but preserve the focal individual/trait.
    subject.x,subject.y=template.x,template.y;subject.energy=20
    template.dependents=[];pup.parent=subject;subject.dependents=[pup]
    world.agents=[subject,pup]
    physical=snapshot(subject)
    physical.pop('impairment');physical.pop('neglect_suffered')
    initial={'parent':physical,'pup':snapshot(pup),'geometry':geometry(world)}
    return cfg,rng,world,pup,initial


def assessment(policy,subject,seed):
    cfg,rng,world,pup,initial=adult_setup(subject,seed)
    history=[];counts=Counter();trait=subject.impairment
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.random.default_generator.manual_seed(seed)
        for tick in range(1,61):
            world.rebuild_occupancy(cfg.crowding_radius)
            movers=[a for a in (subject,pup) if a.alive and a.age+1>=a.dependent_until]
            picks,senses=actions(policy,world,cfg,movers)
            # Measurement-only counterfactual at the SAME live state. No live trait is edited.
            reference=None;actual=None
            if subject in movers:
                actual=np.array(senses[movers.index(subject)])
                reference_parent=copy.copy(subject);reference_parent.impairment=0
                neighbours=world.count_neighbours(subject.x,subject.y,cfg.crowding_radius,exclude=subject)
                reference=np.array(reference_parent.sense(world,cfg,neighbours=neighbours))
                assert np.array_equal(actual[list(OTHER_INPUTS)],reference[list(OTHER_INPUTS)])
                assert np.allclose(actual[list(PUP_INPUTS)],reference[list(PUP_INPUTS)]*(1-trait),rtol=0,atol=1e-12)
                counts['sensory_audit_ticks']+=1
            chosen={}
            for a,pick in zip(movers,picks):
                role='parent' if a is subject else 'pup';chosen[role]=pick
                counts[f'{role}_meals']+=act_and_count_food(a,world,cfg,rng,pick)
            attended=None
            if pup.alive and pup.is_dependent:
                before=pup.neglect_suffered
                act_and_count_food(pup,world,cfg,rng)
                attended=pup.neglect_suffered==before
                counts['care_ticks']+=1;counts['attended_ticks']+=attended
            for a in (subject,pup):cleanup(a,world)
            world.agents=[a for a in (subject,pup) if a.alive]
            world.respawn_step()
            assert subject.impairment==trait
            history.append({'tick':tick,'parent_age':subject.age,'parent_position':[subject.x,subject.y],
                'parent_energy':subject.energy,'parent_alive':subject.alive,'pup_alive':pup.alive,
                'pup_age':pup.age,'pup_neglect_ticks':pup.neglect_ticks,'attended':attended,
                'actions':chosen,'parent_policy_input':actual.tolist() if actual is not None else None,
                'unattenuated_same_state_input':reference.tolist() if reference is not None else None,
                'food_count':len(world.food),'food_hash':digest(sorted(world.food)),
                'focal_population':len(world.agents)})
    mortality={c:sum(not a.alive and a.cause_of_death==c for a in (subject,pup))
               for c in ('starvation','old age','neglect')}
    deaths=sum(mortality.values());assert 2-deaths==len(world.agents)
    return {'seed':seed,'initial':initial,'initial_hash':digest(initial),'impairment':trait,
        'counts':dict(counts),'history':history,'elapsed_ticks':60,
        'independent':pup.alive and not pup.is_dependent,'parent_alive':subject.alive,
        'parent_cause_of_death':subject.cause_of_death,'pup_cause_of_death':pup.cause_of_death,
        'success_after_parent_death':pup.alive and not pup.is_dependent and not subject.alive,
        'starting_population':2,'births':0,'deaths':deaths,'final_population':len(world.agents),
        'deaths_by_cause':mortality,'final_impairment':subject.impairment}


def trajectory(policy,index,disrupted,damage,full=False):
    if index not in (range(20) if full else (0,1,2)):
        raise ValueError('Only the three approved pilot seed triplets or the twenty approved full triplets are allowed')
    before=policy_hash(policy)
    subject,child=childhood(policy,410000+index,disrupted,damage)
    transferred=bridge(policy,subject,420000+index) if subject.alive else None
    adult=assessment(policy,subject,400000+index) if transferred and subject.alive else None
    assert policy_hash(policy)==before
    return {'world_index':index,'disrupted_childhood':disrupted,'damage_enabled':damage,
        'childhood':child,'bridge':transferred,'adult':adult,
        'original_cohort_size':1,'childhood_survivors':int(child['final']['alive']),
        'mature_survivors':int(transferred is not None and transferred['final']['alive']),
        'adults_evaluated':int(adult is not None)}


def summarize(records):
    groups=[]
    for objective in ('survival','offspring'):
        for disrupted in (False,True):
            for damage in (False,True):
                subset=[r for r in records if r['objective']==objective and r['disrupted_childhood']==disrupted and r['damage_enabled']==damage]
                adults=[r['adult'] for r in subset if r['adult'] is not None]
                counts,mortality=Counter(),Counter()
                for a in adults:counts.update(a['counts']);mortality.update(a['deaths_by_cause'])
                groups.append({'objective':objective,'disrupted_childhood':disrupted,'damage_enabled':damage,
                    'assigned_cohort':len(subset),'childhood_survivors':sum(r['childhood_survivors'] for r in subset),
                    'mature_survivors':sum(r['mature_survivors'] for r in subset),'adults_evaluated':len(adults),
                    'independence_successes':sum(a['independent'] for a in adults),
                    'independence_fraction':sum(a['independent'] for a in adults)/len(adults) if adults else None,
                    'cohort_recruitment_fraction':sum(a['independent'] for a in adults)/len(subset) if subset else None,
                    'parent_survivors':sum(a['parent_alive'] for a in adults),'counts':dict(counts),
                    'deaths_by_cause':dict(mortality),'impairments':sorted({a['impairment'] for a in adults})})
    pairs=[]
    for seed in sorted({r['training_seed'] for r in records}):
        effects={}
        for obj in ('survival','offspring'):
            rates={}
            for disrupted in (False,True):
                for damage in (False,True):
                    a=[r['adult'] for r in records if r['training_seed']==seed and r['objective']==obj and r['disrupted_childhood']==disrupted and r['damage_enabled']==damage]
                    rates[(disrupted,damage)]=sum(x['independent'] for x in a)/len(a) if a and all(x is not None for x in a) else None
            effects[obj]=(rates[(True,True)]-rates[(False,True)])-(rates[(True,False)]-rates[(False,False)]) if all(x is not None for x in rates.values()) else None
        interaction=effects['offspring']-effects['survival'] if all(x is not None for x in effects.values()) else None
        pairs.append({'training_seed':seed,**effects,'objective_interaction':interaction})
    return {'groups':groups,'training_pair_effects':pairs,
            'inference':'Training-seed pairs are independent units; matched worlds/branches are repeated evaluations. Missing adult assessments are separate attrition, not adult care failures. Statistical analysis is separate.'}


def validate(records,full=False):
    training_seeds=range(10) if full else (0,1)
    world_indices=range(20) if full else (0,1,2)
    expected=len(training_seeds)*2*len(world_indices)*4
    keys={tuple(r[k] for k in ('training_seed','objective','world_index','disrupted_childhood','damage_enabled')) for r in records}
    gates={'complete_unique_matrix':len(records)==len(keys)==expected,'childhood_survival':True,'mature_survival':True,'exposure_exact':True,
           'trait_exact_and_preserved':True,'controls_identical':True,'bridge_identical':True,
           'initial_assessment_matching':True,'full_followup':True,'food_stock':True,
           'childhood_physics_match':True,'sensory_audit':True,'parent_survival_measured':True}
    for r in records:
        child=r['childhood'];adult=r['adult'];expected=.3 if r['disrupted_childhood'] and r['damage_enabled'] else 0
        gates['childhood_survival'] &= r['childhood_survivors']==1
        gates['mature_survival'] &= r['mature_survivors']==1
        unattended=[h['tick'] for h in child['history'] if h['attended'] is False]
        gates['exposure_exact'] &= child['counts'].get('care_ticks')==59 and child['counts'].get('unattended_ticks')==(18 if r['disrupted_childhood'] else 0) and unattended==(list(range(10,28)) if r['disrupted_childhood'] else []) and child['before_maturation'] is not None and child['before_maturation']['neglect_ticks']==0
        gates['trait_exact_and_preserved'] &= child['final']['impairment']==expected and adult is not None and adult['impairment']==adult['final_impairment']==expected
        gates['full_followup'] &= adult is not None and len(adult['history'])==adult['elapsed_ticks']==60
        gates['sensory_audit'] &= adult is not None and adult['counts'].get('sensory_audit_ticks',0)==sum(h['parent_policy_input'] is not None for h in adult['history'])
        gates['parent_survival_measured'] &= adult is not None and adult['parent_alive']==adult['history'][-1]['parent_alive'] and adult['parent_alive']
        gates['food_stock'] &= all(h['food_count']==60 for h in child['history']) and r['bridge'] is not None and all(h['food_count']==60 for h in r['bridge']['history']) and adult is not None and all(h['food_count']==60 for h in adult['history'])
    for seed in training_seeds:
        for obj in ('survival','offspring'):
            for index in world_indices:
                group=[r for r in records if r['training_seed']==seed and r['objective']==obj and r['world_index']==index]
                controls=[r for r in group if not (r['disrupted_childhood'] and r['damage_enabled'])]
                gates['controls_identical'] &= len(controls)==3 and all(r['adult']==controls[0]['adult'] for r in controls)
                gates['bridge_identical'] &= len(group)==4 and all(r['bridge'] is not None for r in group) and all(r['bridge']['history']==group[0]['bridge']['history'] for r in group)
                gates['initial_assessment_matching'] &= len(group)==4 and all(r['adult'] is not None for r in group) and all(r['adult']['initial_hash']==group[0]['adult']['initial_hash'] for r in group)
                for r in group:
                    gates['childhood_physics_match'] &= r['childhood']['initial']==group[0]['childhood']['initial']
                    for a,b in zip(r['childhood']['history'],group[0]['childhood']['history']):
                        gates['childhood_physics_match'] &= a['caregiver']==b['caregiver'] and a['food_hash']==b['food_hash']
                        gates['childhood_physics_match'] &= all(a['subject'][k]==b['subject'][k] for k in ('position','age','energy','lifespan','alive','parent_attached','dependents'))
    for index in world_indices:
        group=[r for r in records if r['world_index']==index]
        gates['initial_assessment_matching'] &= len(group)==len(training_seeds)*8 and all(r['adult'] is not None for r in group) and len({r['adult']['initial_hash'] for r in group})==1
    return gates


def execute(out,full=False):
    training_seeds=range(10) if full else (0,1)
    world_indices=range(20) if full else (0,1,2)
    expected=len(training_seeds)*2*len(world_indices)*4
    out=Path(out).resolve()
    if out==ROOT/'results' or any(p.is_dir() and (p==out or p in out.parents) for p in (ROOT/'results').iterdir()):
        raise ValueError('Use a separate new Phase 3 directory')
    checkpoints=ROOT/'results/final_reward_v1'
    hashes={p.name:file_hash(p) for p in checkpoints.glob('*.pt')}
    policies={}
    for seed in training_seeds:
        for obj in ('survival','offspring'):
            p=PolicyNet(seed=seed);p.load_state_dict(torch.load(checkpoints/f'{obj}_s{seed}.pt',weights_only=True));p.eval()
            for parameter in p.parameters():
                parameter.requires_grad_(False);assert torch.isfinite(parameter).all()
            trained_hash_expected=json.loads((checkpoints/f'{obj}_s{seed}_evaluation.json').read_text())['trained_policy_hash']
            assert policy_hash(p)==trained_hash_expected;policies[(seed,obj)]=p
    out.mkdir(parents=True,exist_ok=False)
    cfg=asdict(config());cfg['reproduce_threshold']='disabled (+infinity)'
    manifest={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'runner_sha256':file_hash(Path(__file__)),'checkpoint_sha256':hashes,'config':cfg,
        'training_seeds':list(training_seeds),'adult_world_seeds':[400000+i for i in world_indices],
        'childhood_world_seeds':[410000+i for i in world_indices],'bridge_world_seeds':[420000+i for i in world_indices],
        'unattended_age_interval':[10,27],'childhood_background':8,'caregiver_distance':2,
        'caregiver_energy':60,'assessment_energy':20,'stages_ticks':[60,60,60],
        'expected_trajectories':expected,'scope':'full' if full else 'pilot','purpose':__doc__,'torch':torch.__version__,'numpy':np.__version__,
        'sampling':'Separate seeded CPU Torch fork and local Python RNG for every stage; no optimizer'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    records=[];start=time.perf_counter()
    for seed in training_seeds:
        for obj in ('survival','offspring'):
            for index in world_indices:
                for disrupted in (False,True):
                    for damage in (False,True):
                        r=trajectory(policies[(seed,obj)],index,disrupted,damage,full=full)
                        r.update(training_seed=seed,objective=obj);records.append(r)
                        name=f'{obj}_s{seed}_w{400000+index}_c{int(disrupted)}_d{int(damage)}.json'
                        (out/name).write_text(json.dumps(r,indent=2))
            print(obj,seed,'completed',len(world_indices)*4,'trajectories',flush=True)
    assert all(file_hash(checkpoints/name)==h for name,h in hashes.items())
    summary=summarize(records);summary.update(trajectories=len(records),expected_trajectories=expected,
        total_seconds=time.perf_counter()-start,technical_gates=validate(records,full=full),
        checkpoint_integrity_verified=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    parser.add_argument('--full',action='store_true',help='Approved 1,600-trajectory matrix; assay parameters unchanged')
    torch.set_num_threads(1)
    args=parser.parse_args();execute(args.out,full=args.full)
