"""Regressions for evaluation-only initial policies and three-channel ablation."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from experiments import neural_policy_evaluation as evaluation
from experiments.final_rewards import assigned_pup_trial, state_hash
from rl import PolicyNet


@pytest.mark.parametrize('seed', range(10))
def test_exact_original_initialization_for_both_objectives(seed):
    initial = state_hash(PolicyNet(seed=seed))
    for objective in ('survival', 'offspring'):
        saved = json.loads((evaluation.SOURCE/f'{objective}_s{seed}_evaluation.json').read_text())
        assert initial == saved['initial_policy_hash']


def test_initialization_mismatch_stops_before_evaluation(tmp_path):
    (tmp_path/'survival_s0_evaluation.json').write_text(json.dumps({'initial_policy_hash': 'wrong'}))
    with pytest.raises(ValueError, match='initialization mismatch'):
        evaluation.load_policies([0], tmp_path)


@pytest.mark.parametrize('masked', [False, True])
def test_masking_only_own_pup_inputs_without_mutating_caller_or_weights(masked):
    policy = PolicyNet(seed=5)
    before = state_hash(policy)
    inputs = torch.linspace(-1, 1, 36).reshape(3, 12)
    original = inputs.clone()
    observed = evaluation.ObservedPolicy(policy, masked)
    with torch.no_grad(): logits = observed(inputs)
    reference = inputs.clone()
    if masked: reference[:, 9:] = 0
    assert torch.equal(logits, policy(reference))
    assert torch.equal(original, inputs) and state_hash(policy) == before
    for audit in observed.observations:
        assert audit['raw_input'][:9] == audit['policy_input'][:9]
        assert audit['policy_input'][9:] == ([0., 0., 0.] if masked else audit['raw_input'][9:])
        if not masked:
            assert audit['normal_to_applied_kl'] == 0 and not audit['argmax_changed']


@pytest.mark.parametrize('objective', ['survival', 'offspring'])
@pytest.mark.parametrize('energy', [10, 20])
def test_noop_exactly_replays_saved_outcomes_and_matches_masked_start(objective, energy):
    policies, _ = evaluation.load_policies([0])
    policy = policies[(0, objective)]; before = state_hash(policy)
    assert not policy.training and all(not p.requires_grad for p in policy.parameters())
    original = assigned_pup_trial(policy, 100000, energy)
    normal = evaluation.trial(policy, 100000, energy)
    masked = evaluation.trial(policy, 100000, energy, True)
    saved = json.loads((evaluation.SOURCE/f'{objective}_s0_evaluation.json').read_text())
    expected = next(t for t in saved['assigned_pup_trials'] if t['seed'] == 100000 and t['initial_energy'] == energy)
    assert normal['outcome'] == original
    assert json.loads(json.dumps(original)) == expected
    assert normal['outcome']['initial_conditions'] == masked['outcome']['initial_conditions']
    assert normal['input_audits'][0]['raw_input'] == masked['input_audits'][0]['raw_input']
    assert state_hash(policy) == before
    for result in (normal, masked):
        o = result['outcome']
        assert o['starting_population']+o['births']-o['deaths'] == o['final_population']
        assert o['parent_outcome'] == ('dead' if not o['parent_alive'] else 'survived_full_trial' if o['elapsed_ticks']==60 else 'alive_at_early_stop')


def test_deterministic_replay_and_global_rng_isolation():
    policy = PolicyNet(seed=1)
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    py, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    first = evaluation.trial(policy, 100001, 20, True)
    assert random.getstate() == py
    now = np.random.get_state()
    assert now[0] == numpy[0] and np.array_equal(now[1], numpy[1]) and now[2:] == numpy[2:]
    assert torch.equal(cpu, torch.get_rng_state())
    random.random(); np.random.random(37); torch.rand(41)
    assert first == evaluation.trial(policy, 100001, 20, True)


def test_existing_or_nested_result_directory_is_rejected(tmp_path):
    with pytest.raises(FileExistsError): evaluation.execute(tmp_path)
    with pytest.raises(ValueError, match='nest output'):
        evaluation.execute(evaluation.SOURCE/'must_not_create')
