"""Budgeted training, resumable checkpoints, and append-only JSONL evidence."""
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from config import SimConfig
from . import ENVIRONMENT_VERSION
from .data import RewardConfig
from .environment import collect
from .evaluation import SeedBlocks, evaluate, summarize
from .learning import Learner, LearnerConfig


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = list(root.glob('*.py')) + list((root/'v2').glob('*.py'))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def provenance():
    root = Path(__file__).resolve().parents[2]
    return dict(environment_version=ENVIRONMENT_VERSION, python=platform.python_version(),
                torch=str(torch.__version__), numpy=np.__version__, source_sha256=source_hashes(),
                git_commit=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                device='cpu', torch_threads=torch.get_num_threads())


class Experiment:
    def __init__(self, cfg, learner_config=LearnerConfig(), rewards=RewardConfig(),
                 seeds=SeedBlocks(), validation_every=1, *, validation_decisions=None,
                 assigned_care=False, care_energies=(10, 20)):
        if validation_every < 0 or learner_config.seed < 0:
            raise ValueError('nonnegative validation frequency and training seed required')
        if learner_config.seed * 10000 >= seeds.train_size:
            raise ValueError('training replicate is outside seed block')
        if rewards.objective == 'offspring' and (not cfg.parental_care_enabled or cfg.dependency_ticks <= 0):
            raise ValueError('offspring-independence reward requires positive dependency with parental care')
        if validation_decisions is not None and validation_decisions <= 0:
            raise ValueError('decision validation interval must be positive')
        self.validation_decisions_interval = validation_decisions
        self.next_validation_decision = validation_decisions
        self.assigned_care = assigned_care
        self.care_energies = tuple(care_energies)
        self.validation_history = []
        self.cfg, self.rewards, self.seeds = cfg, rewards, seeds
        self.learner = Learner(learner_config)
        self.validation_every = validation_every
        self.episode_index = self.training_decisions = self.training_ticks = self.validation_decisions = 0
        self.history = []

    def step(self):
        if self.episode_index >= 10000:
            raise ValueError('replicate seed allocation exhausted')
        seed = self.seeds.train(self.learner.config.seed * 10000 + self.episode_index)
        started = time.perf_counter()
        episode = collect(self.cfg, self.learner.policy, seed, self.rewards)
        diagnostics = self.learner.update([episode])
        seconds = time.perf_counter() - started
        self.training_decisions += episode.decisions
        self.training_ticks += episode.metrics['elapsed_ticks']
        self.episode_index += 1
        record = dict(episode=self.episode_index, world_seed=seed, **diagnostics,
                      training_decisions=self.training_decisions, training_world_ticks=self.training_ticks,
                      discounted_colony_return=sum(self.learner.config.gamma**e.tick * e.amount for e in episode.events),
                      episode_metrics=episode.metrics, training_seconds=seconds,
                      checkpoint_id=f'episode-{self.episode_index:06d}.pt', validation=None)
        if self.validation_decisions_interval is not None:
            crossed = []
            while self.training_decisions >= self.next_validation_decision:
                crossed.append(self.next_validation_decision)
                self.next_validation_decision += self.validation_decisions_interval
            if crossed:
                record['validation'] = self.validate(crossed)
        elif self.validation_every and self.episode_index % self.validation_every == 0:
            # Phase 1 API compatibility only; approved pilots use decision thresholds.
            record['validation'] = self.validate([])
        record['validation_decisions'] = self.validation_decisions
        self.history.append(record)
        return record

    def validate(self, thresholds):
        """One frozen panel per boundary, shared if several thresholds were crossed."""
        from .care import evaluate_assigned
        records = evaluate(self.cfg, self.learner.policy, self.seeds, rewards=self.rewards)
        assigned = (evaluate_assigned(self.cfg, self.learner.policy, self.seeds, self.care_energies)
                    if self.assigned_care else None)
        natural_cost = sum(r['adult_decisions'] for r in records)
        assigned_cost = assigned['summary']['adult_decisions'] if assigned else 0
        self.validation_decisions += natural_cost + assigned_cost
        result = dict(thresholds=list(thresholds), actual_training_decisions=self.training_decisions,
                      optimizer_updates=self.learner.updates, summary=summarize(records), worlds=records,
                      assigned=assigned, natural_validation_decisions=natural_cost,
                      assigned_validation_decisions=assigned_cost,
                      cumulative_validation_decisions=self.validation_decisions)
        self.validation_history.append(result)
        return result

    def validate_initial(self):
        if self.episode_index != 0 or self.validation_history:
            raise ValueError('initial validation must run exactly once before training')
        return self.validate([0])

    def train_to_budget(self, decisions, max_episodes, *, max_overshoot=None):
        """Whole episodes only; report bounded overshoot rather than truncate MC.

        The final overshoot is < horizon * max_population decisions. Equal target
        budgets do not imply exactly equal realized budgets; report both. Validation
        is entirely excluded from training counters and does not alter action RNGs.
        """
        if decisions <= self.training_decisions or max_episodes <= 0:
            raise ValueError('budget must increase, with a positive episode guard')
        for _ in range(max_episodes):
            self.step()
            if self.training_decisions >= decisions:
                if max_overshoot is not None and self.training_decisions - decisions > max_overshoot:
                    raise RuntimeError('prespecified decision-budget overshoot exceeded')
                return dict(target=decisions, realized=self.training_decisions,
                            overshoot=self.training_decisions-decisions)
        raise RuntimeError('episode guard reached before budget; no silent partial campaign')

    def checkpoint(self, path):
        path = Path(path)
        payload = dict(schema=2, provenance=provenance(), config=asdict(self.cfg),
                       rewards=asdict(self.rewards), seeds=asdict(self.seeds),
                       validation_every=self.validation_every, learner=self.learner.state(),
                       validation_decisions_interval=self.validation_decisions_interval,
                       next_validation_decision=self.next_validation_decision,
                       assigned_care=self.assigned_care, care_energies=self.care_energies,
                       validation_history=self.validation_history,
                       episode_index=self.episode_index, training_decisions=self.training_decisions,
                       training_ticks=self.training_ticks, validation_decisions=self.validation_decisions,
                       history=self.history)
        # Exclusive creation prevents silent loss of an earlier checkpoint.
        with path.open('xb') as stream:
            torch.save(payload, stream)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def restore(cls, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['schema'] != 2 or data['provenance']['environment_version'] != ENVIRONMENT_VERSION:
            raise ValueError('checkpoint schema/environment mismatch')
        current = provenance()
        for key in ('python', 'torch', 'numpy', 'source_sha256', 'device', 'torch_threads'):
            if current[key] != data['provenance'][key]:
                raise ValueError(f'checkpoint runtime/source mismatch: {key}')
        blocks = data['seeds']
        blocks['validation'], blocks['final_test'] = tuple(blocks['validation']), tuple(blocks['final_test'])
        result = cls(SimConfig(**data['config']), LearnerConfig(**data['learner']['config']),
                     RewardConfig(**data['rewards']), SeedBlocks(**blocks), data['validation_every'],
                     validation_decisions=data['validation_decisions_interval'],
                     assigned_care=data['assigned_care'], care_energies=data['care_energies'])
        result.learner = Learner.restore(data['learner'])
        for key in ('episode_index', 'training_decisions', 'training_ticks', 'validation_decisions', 'history',
                    'next_validation_decision', 'validation_history'):
            setattr(result, key, data[key])
        return result


def run_smoke(output, architecture='original', algorithm='reinforce', objective='offspring'):
    """Intentionally bounded CLI workflow, not a research campaign."""
    from config import DEFAULT
    output = Path(output).resolve()
    protected = Path(__file__).resolve().parents[2]/'results'
    if output == protected or protected in output.parents:
        raise ValueError('Phase 1 outputs must be outside preserved results/')
    output.mkdir(parents=True, exist_ok=False)
    cfg = DEFAULT.variant('v2-technical-smoke', width=12, height=12, n_ticks=16,
                          n_food=20, n_initial_agents=4, max_population=12,
                          parental_care_enabled=True, dependency_ticks=4)
    experiment = Experiment(cfg, LearnerConfig(architecture=architecture, algorithm=algorithm),
                            RewardConfig(objective))
    manifest = dict(schema=1, technical_smoke_only=True, config=asdict(cfg),
                    learner=asdict(experiment.learner.config), reward=asdict(experiment.rewards),
                    seeds=asdict(experiment.seeds), provenance=provenance(),
                    final_test_executed=False)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    with (output/'training.jsonl').open('x') as stream:
        for _ in range(2):
            record = experiment.step()
            checksum = experiment.checkpoint(output/record['checkpoint_id'])
            record = {**record, 'checkpoint_sha256': checksum}
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            stream.flush()
    return experiment
