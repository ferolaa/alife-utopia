"""Frozen matched-world evaluation with distinct train/validation/final blocks."""
from dataclasses import dataclass
import hashlib

import torch

from .data import RewardConfig
from .environment import collect


@dataclass(frozen=True)
class SeedBlocks:
    # New namespace, separate from legacy seeds 0..9119 and 100000..100019.
    train_start: int = 1_000_000
    train_size: int = 1_000_000
    validation: tuple = (3_000_000, 3_000_001, 3_000_002)
    final_test: tuple = tuple(range(4_000_000, 4_000_020))

    def __post_init__(self):
        if self.train_start < 0 or self.train_size <= 0 or not self.validation or not self.final_test:
            raise ValueError('nonempty nonnegative seed blocks required')
        lo, hi = self.train_start, self.train_start + self.train_size
        held = self.validation + self.final_test
        if any(s < 0 or lo <= s < hi for s in held) or len(set(held)) != len(held):
            raise ValueError('seed blocks overlap or contain duplicate/negative seeds')

    def train(self, index):
        if not 0 <= index < self.train_size:
            raise ValueError('training seed block exhausted')
        return self.train_start + index


def state_hash(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def evaluate(cfg, policy, seeds=SeedBlocks(), *, split='validation',
             rewards=RewardConfig(), allow_final=False):
    if split not in ('validation', 'final_test'):
        raise ValueError('evaluation split must be validation or final_test')
    if split == 'final_test' and not allow_final:
        raise ValueError('final test is sealed; use only after protocol approval')
    before = state_hash(policy)
    mode = policy.training
    try:
        policy.eval()
        with torch.no_grad():
            episodes = [collect(cfg, policy, s, rewards, action_seed=s)
                        for s in getattr(seeds, split)]
    finally:
        policy.train(mode)
    if state_hash(policy) != before:
        raise RuntimeError('evaluation modified policy state')
    return [dict(world_seed=e.world_seed, action_seed=e.action_seed,
                 initial_hash=e.initial_hash, policy_hash=before, **e.metrics) for e in episodes]


def summarize(records):
    """Exposure-weighted descriptive rates; zeros in denominator are undefined."""
    totals = {k: sum(r[k] for r in records) for k in
              ('births', 'eligible_births', 'independent_offspring', 'care_ticks', 'attended_ticks',
               'adult_decisions', 'surviving_adult_ticks', 'elapsed_ticks', 'total_reward')}
    # Independent successes counted in the eligible panel only, not late births.
    totals['eligible_independent_offspring'] = sum(r['eligible_independent_offspring'] for r in records)
    totals['attendance_fraction'] = (totals['attended_ticks']/totals['care_ticks']
                                     if totals['care_ticks'] else None)
    totals['independence_fraction'] = (totals['eligible_independent_offspring']/totals['eligible_births']
                                       if totals['eligible_births'] else None)
    totals['episodes'] = len(records)
    return totals
