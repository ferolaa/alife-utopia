# Portable frozen neural-policy verification and reanalysis

Run from repository root after the COMPLETE proposed atomic inventory is present.
Use Python 3.12.14 and requirements-reproduction.txt CPU pins. See the original
REPRODUCIBILITY.md for environment caveats. Fresh environment installation has not
been tested. Use ordinary Python, not -O: scientific checks use assertions.
Every destination below must be NEW. Restoration needs about 100 MB disk for the full
original JSON dataset, plus memory for parsed records and temporary analysis.

```bash
python3.12 -m venv .venv-repro
.venv-repro/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-reproduction.txt
.venv-repro/bin/python -m pytest -q
mkdir -p work
.venv-repro/bin/python experiments/archive_neural_policy.py verify results/neural_policy_v1_archive/records.zip results/neural_policy_v1_archive/ARCHIVE.json
.venv-repro/bin/python experiments/archive_neural_policy.py restore results/neural_policy_v1_archive/records.zip results/neural_policy_v1_archive/ARCHIVE.json work/neural_raw_new
.venv-repro/bin/python experiments/verify_neural_policy.py --data work/neural_raw_new --replay --out work/neural_verification_new.json
.venv-repro/bin/python experiments/verify_neural_policy_responses.py --data work/neural_raw_new --out work/neural_response_verification_new.json
.venv-repro/bin/python experiments/analyze_neural_policy.py --data work/neural_raw_new --out work/neural_analysis_new
.venv-repro/bin/python experiments/archive_neural_policy.py restore-pilot work/neural_raw_new results/neural_policy_pilot_v1 work/neural_pilot_raw_new
```

The original full records.json is excluded from Git. Override --data with the restored
path as shown; do not expect it to exist in the committed metadata-only raw directory.
Restoration verifies archive SHA-256, member names, sizes and all original-byte hashes.
The full archive includes manifest, records, summary and INTEGRITY exactly as saved.
The pilot subset is restored with its original serialization and compared with its
preserved SHA-256; no rerunning or training is necessary to recover it.

Verification checks every trial key, initial-parameter hashes against BOTH training
objectives, checkpoint hashes,800saved normal outcomes, same-state mask channels,
population/deaths, training-seed effects, confidence intervals/exact tests/Holm, and
optionally 100 frozen replays. Response verification independently recalculates all
71,024 normal/applied probabilities and KL values from frozen parameters. Neither
command fits weights. Analyzer creates statistics, per-seed CSV, concise report,
caption, vector PDF and 300-dpi PNG. No simulation is performed by analysis. Repeated
analysis reproduces JSON,PNG,CSV and report text; PDF creation dates may vary.

For exact pilot-source provenance see neural_policy_v1_analysis/pilot_runner_source.txt.
The current evaluator extends pilot scope to the approved full matrix and adds cause
summaries; it preserves the same delegated original dynamics and trial implementation.
An authorized future full reproduction (not necessary during preservation) would use:

```bash
.venv-repro/bin/python experiments/neural_policy_evaluation.py --full --out results/neural_reproduction_new
```

This is evaluation only using already saved checkpoints, not retraining. No full
rerun was performed for packaging. Existing Phases 1–3 commands and data remain intact.
