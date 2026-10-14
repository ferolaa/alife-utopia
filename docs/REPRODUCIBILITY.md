# Reproducing completed Phases 1 and 2

Run commands from the repository root. No command below overwrites a completed dataset.
Runners and report generators refuse existing output directories. Do not use the original
result directories as reproduction targets. Core mechanics were at commit
`e9454128c506978f3154486c902120296691efdf`; each original manifest records the runner SHA.
The preserved runner bytes still match those hashes. New preservation scripts do not
change the original runners or training.

## Environment

Recorded environment: Python 3.12.14, NumPy 2.5.4, PyTorch 2.14.1+cpu,
Matplotlib 3.11.2, pytest 9.1.1. Full installed versions are in
`requirements-reproduction.txt` and `results/preservation_v1/ENVIRONMENT.json`.
The complete RL tests require PyTorch. Use CPU with one Torch intra-op thread,
as configured by the experiment CLIs. CUDA and multithreaded numerical equality are
not claimed. Installation on a new machine has not been tested; do not silently replace
an unavailable pin and then claim byte-identical reproduction.

```bash
python3.12 -m venv .venv-repro
.venv-repro/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-reproduction.txt
.venv-repro/bin/python -m pytest -q
```

The commands below use the project's existing `.venv`. In a clean checkout using the
new environment above, substitute `.venv-repro/bin/python` for `.venv/bin/python`.

## Verify existing data without retraining

```bash
.venv/bin/python experiments/verify_completed.py
.venv/bin/python experiments/verify_completed.py --replay --out verification_new.json
.venv/bin/python experiments/analyze_controlled_care.py --out results/controlled_care_reanalysis_new
.venv/bin/python experiments/report_completed.py --out results/report_figures_new
```

The independent verifier recomputes both phases' denominators, bootstrap intervals,
exact sign-flip tests, Phase 2 Holm adjustments, initial matching, mortality accounting,
checkpoint hashes and finite weights. `--replay` repeats 20 Phase 1 assigned-pup trials,
20 Phase 2 trials and two Phase 1 natural episodes without learning; all 42 matched.
Phase 2 reanalysis reproduces all saved groups and 27 paired contrasts exactly.

The gzip archive is authoritative for committed Phase 2 raw trials. Its receipts verify
both compressed bytes and decompressed original JSON bytes. The analysis and verification
scripts transparently load these records if local uncompressed originals are absent.
Keep `results/controlled_care_v1/manifest.json` and `summary.json` at their original paths.
Original full JSON remains untouched locally and is deliberately excluded from Git to
avoid duplicating 157.5 MB. All 96 pilot trial records match full-run records exactly;
pilot manifest and summary preserve its provenance without duplicate trial records.

## Re-run approved experiments into new directories (not performed during preservation)

Phase 1 requires retraining and was about 34 minutes in this environment. This command
documents reproduction only; retraining requires separate authorization.

```bash
.venv/bin/python experiments/final_rewards.py --out results/final_reward_reproduction_new --seeds 0,1,2,3,4,5,6,7,8,9 --iterations 120 --ticks 500 --eval-seeds 100000,100001,100002,100003,100004,100005,100006,100007,100008,100009,100010,100011,100012,100013,100014,100015,100016,100017,100018,100019 --eval-ticks 1000
```

Phase 2 loads the original twenty frozen checkpoints. Recorded runtime: 77.89 seconds.
The first three worlds below are also pilot worlds; all twenty are disjoint from training.

```bash
.venv/bin/python experiments/controlled_care.py --out results/controlled_care_reproduction_new --seeds 0,1,2,3,4,5,6,7,8,9 --world-seeds 300000,300001,300002,300003,300004,300005,300006,300007,300008,300009,300010,300011,300012,300013,300014,300015,300016,300017,300018,300019
```

## Analysis conventions

Phase 1 bootstrap: 10,000 draws; Phase 2: 100,000; both RNG seed 20261014.
Only independent training pairs are resampled. World seeds and energies are matched
within each pair. Exact tests enumerate 1,024 sign flips. Phase 2 Holm corrections
apply separately to pooled and energy-specific nine-contrast families. Intervals are
unadjusted percentile intervals conditional on the fixed world panel, not simultaneous
intervals or uncertainty over a new world distribution.

Archive regeneration, if needed, also requires a new target:

```bash
.venv/bin/python experiments/archive_controlled_care.py --out results/controlled_care_archive_new
```

Legacy data, checkpoints and earlier outputs remain preserved. No `/workspace/work`
file or absolute workspace path is required for the proposed committed reproducer.
