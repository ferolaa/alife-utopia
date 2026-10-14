"""Evaluation-only initial-network comparison and own-pup input ablation.

Reuse the historical assigned-pup protocol verbatim. No optimizer or core edits.
Normal trained trials are shared across experiments A/B, never duplicate replicates.
"""
import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
import torch
from rl import PolicyNet
from experiments.final_rewards import assigned_pup_trial, state_hash

SOURCE = ROOT / 'results/final_reward_v1'
CELLS = ('initial', 'survival_normal', 'offspring_normal', 'survival_masked', 'offspring_masked')
CONTRASTS = {
    'survival_training': ('survival_normal', 'initial'),
    'offspring_training': ('offspring_normal', 'initial'),
    'survival_masking': ('survival_masked', 'survival_normal'),
    'offspring_masking': ('offspring_masked', 'offspring_normal'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policies(seeds, source=SOURCE):
    policies, hashes = {}, {}
    for seed in seeds:
        initial = PolicyNet(seed=seed)
        h = state_hash(initial)
        for obj in ('survival', 'offspring'):
            saved = json.loads((source / f'{obj}_s{seed}_evaluation.json').read_text())
            if h != saved['initial_policy_hash']:
                raise ValueError(f'Exact initialization mismatch: {obj}, seed {seed}')
            checkpoint = source / f'{obj}_s{seed}.pt'
            trained = PolicyNet(seed=seed)
            trained.load_state_dict(torch.load(checkpoint, weights_only=True))
            if state_hash(trained) != saved['trained_policy_hash']:
                raise ValueError(f'Trained parameter mismatch: {obj}, seed {seed}')
            policies[(seed, obj)] = trained
            hashes[checkpoint.name] = sha(checkpoint)
        policies[(seed, 'initial')] = initial
    initial_hashes = [state_hash(policies[(s, 'initial')]) for s in seeds]
    if len(set(initial_hashes)) != len(seeds):
        raise ValueError('Initial networks are not distinct')
    for policy in policies.values():
        policy.eval()
        for parameter in policy.parameters():
            if not torch.isfinite(parameter).all():
                raise ValueError('Nonfinite checkpoint')
            parameter.requires_grad_(False)
    return policies, hashes


class ObservedPolicy(torch.nn.Module):
    """Intervene only on the three last input channels, on a cloned tensor.

    Same-state reference probabilities are computed without sampling or RNG draws.
    They quantify local response sensitivity, not hidden representation semantics.
    """
    def __init__(self, policy, masked=False):
        super().__init__()
        self.policy, self.masked = policy, masked
        self.observations = []

    def forward(self, senses):
        if senses.shape[-1] != 12:
            raise ValueError('Expected original twelve-input architecture')
        used = senses.clone()
        if self.masked:
            used[..., 9:12] = 0
        reference = self.policy(senses)
        logits = self.policy(used) if self.masked else reference
        for raw, applied, normal, actual in zip(senses, used, reference, logits):
            p, q = torch.softmax(normal, -1), torch.softmax(actual, -1)
            self.observations.append({
                'raw_input': raw.tolist(), 'policy_input': applied.tolist(),
                'normal_probabilities': p.tolist(), 'applied_probabilities': q.tolist(),
                'normal_to_applied_kl': float((p * (torch.log_softmax(normal, -1) - torch.log_softmax(actual, -1))).sum()),
                'argmax_changed': bool(normal.argmax() != actual.argmax()),
            })
        return logits


def trial(policy, seed, energy, masked=False):
    before = state_hash(policy)
    observed = ObservedPolicy(policy, masked)
    outcome = assigned_pup_trial(observed, seed, energy)
    if state_hash(policy) != before:
        raise ValueError('Evaluation changed weights')
    assert outcome['starting_population'] + outcome['births'] - outcome['deaths'] == outcome['final_population']
    assert outcome['care_ticks'] >= outcome['attended_ticks'] >= 0
    assert outcome['deaths'] == int(outcome['parent_cause_of_death'] is not None) + int(outcome['pup_cause_of_death'] is not None)
    for audit in observed.observations:
        assert audit['raw_input'][:9] == audit['policy_input'][:9]
        assert audit['policy_input'][9:] == ([0., 0., 0.] if masked else audit['raw_input'][9:])
    return {'outcome': outcome, 'input_audits': observed.observations}


def summarize(records, seeds, worlds):
    groups = []
    for energy in (None, 10, 20):
        for cell in CELLS:
            rows = [r for r in records if r['cell'] == cell and (energy is None or r['energy'] == energy)]
            outcomes = [r['outcome'] for r in rows]
            care = sum(o['care_ticks'] for o in outcomes)
            audits = [a for r in rows for a in r['input_audits']]
            groups.append({'cell': cell, 'energy': energy, 'trials': len(rows),
                'successes': sum(o['independent'] for o in outcomes),
                'success_rate': sum(o['independent'] for o in outcomes)/len(rows),
                'attended_ticks': sum(o['attended_ticks'] for o in outcomes), 'care_ticks': care,
                'attendance_fraction': sum(o['attended_ticks'] for o in outcomes)/care if care else None,
                'parent_meals': sum(o['parent_meals'] for o in outcomes),
                'parent_deaths': sum(not o['parent_alive'] for o in outcomes),
                'parent_deaths_by_cause': {cause: sum(o['parent_cause_of_death'] == cause for o in outcomes) for cause in ('starvation', 'old age', 'neglect')},
                'pup_deaths_by_cause': {cause: sum(o['pup_cause_of_death'] == cause for o in outcomes) for cause in ('starvation', 'old age', 'neglect')},
                'parent_alive_early_stop': sum(o['parent_outcome'] == 'alive_at_early_stop' for o in outcomes),
                'parent_survived_full_trial': sum(o['parent_outcome'] == 'survived_full_trial' for o in outcomes),
                'posthumous_independence': sum(o['independent'] and not o['parent_alive'] for o in outcomes),
                'mean_same_state_kl': float(np.mean([a['normal_to_applied_kl'] for a in audits])),
                'argmax_change_fraction': float(np.mean([a['argmax_changed'] for a in audits]))})
    effects = {}
    for name, (a, b) in CONTRASTS.items():
        effects[name] = [
            sum(r['outcome']['independent'] for r in records if r['training_seed'] == s and r['cell'] == a)/(2*len(worlds)) -
            sum(r['outcome']['independent'] for r in records if r['training_seed'] == s and r['cell'] == b)/(2*len(worlds))
            for s in seeds]
    return {'groups': groups, 'seed_effects': effects,
            'mean_effects': {n: float(np.mean(v)) for n, v in effects.items()},
            'pilot_inference': 'Descriptive only; two training seeds do not support robust inference.' if len(seeds) == 2 else None}


def execute(out, full=False):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Output must be a NEW directory')
    results = ROOT / 'results'
    if any(p.is_dir() and (out == p or p in out.parents) for p in results.iterdir()):
        raise ValueError('Do not nest output inside an existing experiment')
    torch.set_num_threads(1)
    seeds = list(range(10)) if full else [0, 1]
    worlds = list(range(100000, 100020 if full else 100003))
    expected = len(seeds) * len(worlds) * 2 * len(CELLS)
    # Verify ALL ten initializations before any pilot trial, not just the subset.
    policies, checkpoints = load_policies(list(range(10)))
    out.mkdir(parents=True)
    manifest = {'scope': 'full prespecified evaluation' if full else 'pilot only', 'training_seeds': seeds, 'world_seeds': worlds,
        'parent_energies': [10, 20], 'cells': CELLS, 'expected_trials': expected,
        'unique_initial_networks_verified': 10, 'unique_initial_networks_evaluated': len(seeds),
        'trained_networks_evaluated': 2*len(seeds), 'checkpoint_sha256': checkpoints,
        'initial_parameter_hashes': {str(s): state_hash(policies[(s, 'initial')]) for s in range(10)},
        'runner_sha256': sha(__file__), 'original_evaluator_sha256': sha(ROOT/'experiments/final_rewards.py'),
        'base_commit': '3c6a601a7517c20780b0725d77d3663beba2bc5c',
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
        'planned_full_contrasts': CONTRASTS,
        'limitations': ['Original early stop after pup death; parent follow-up censored.',
                        'Masking is an input-distribution shift; no neural representation claim.',
                        'Original parental-care timing and reward-credit limitations unchanged.']}
    records = []; start = time.perf_counter()
    for s, w, e in itertools.product(seeds, worlds, (10, 20)):
        initial = None
        for cell in CELLS:
            obj = cell.split('_')[0]; masked = cell.endswith('masked')
            result = trial(policies[(s, obj)], w, e, masked)
            if initial is None: initial = result['outcome']['initial_conditions']
            assert result['outcome']['initial_conditions'] == initial
            if obj != 'initial' and not masked:
                saved = json.loads((SOURCE/f'{obj}_s{s}_evaluation.json').read_text())
                original = next(t for t in saved['assigned_pup_trials'] if t['seed'] == w and t['initial_energy'] == e)
                assert json.loads(json.dumps(result['outcome'])) == original, 'No-op fails exact historical replay'
            records.append({'training_seed': s, 'world_seed': w, 'energy': e, 'cell': cell, **result})
    assert len(records) == expected
    manifest['evaluation_seconds'] = time.perf_counter() - start
    summary = summarize(records, seeds, worlds)
    for name, value in [('manifest', manifest), ('records', records), ('summary', summary)]:
        (out/f'{name}.json').write_text(json.dumps(value, indent=2)+'\n')
    hashes = {p.name: sha(p) for p in out.iterdir()}
    (out/'INTEGRITY.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps({'seconds': manifest['evaluation_seconds'], 'trials': len(records), 'effects': summary['mean_effects']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    execute(args.out, args.full)
