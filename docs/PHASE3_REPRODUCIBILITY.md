# Portable Phase 3 reanalysis, no learning or new scientific experiment

Run from a fresh checkout containing the complete Phase 3 atomic inclusion inventory.
Use Python 3.12.14 and the recorded CPU environment (see docs/REPRODUCIBILITY.md).
Installation on a new machine has not been tested; equality is verified in the pinned
existing environment. All target directories must be new. Allow about 650 MB temporary
space for restored files plus parsed records/reanalysis working memory.

```bash
python3.12 -m venv .venv-repro
.venv-repro/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-reproduction.txt
.venv-repro/bin/python -m pytest -q
.venv-repro/bin/python experiments/archive_developmental_care.py verify results/developmental_care_v1_archive/records.zip results/developmental_care_v1_archive/ARCHIVE.json
.venv-repro/bin/python experiments/archive_developmental_care.py restore results/developmental_care_v1_archive/records.zip results/developmental_care_v1_archive/ARCHIVE.json work/phase3_raw_new
.venv-repro/bin/python experiments/verify_developmental_care.py --data work/phase3_raw_new --replay --out work/phase3_verification_new.json
.venv-repro/bin/python experiments/analyze_developmental_care.py --data work/phase3_raw_new --out work/phase3_analysis_new
.venv-repro/bin/python experiments/archive_developmental_care.py restore results/developmental_care_pilot_v1_archive/records.zip results/developmental_care_pilot_v1_archive/ARCHIVE.json work/phase3_pilot_raw_new
```

Archives contain original manifest/summary and every original trajectory, including
historical incorrect expected-count metadata. Restoration validates archive SHA-256,
exact member inventory, original lengths and every record SHA-256; it refuses existing
destinations and rejects non-flat JSON members. Use ordinary Python (not `python -O`),
as the existing verifier and packaging checks use assertions.

Verifier additionally validates METADATA_CORRECTION.json, original executed source,
the actual 1,600 count, controls, population accounting, mortality, input attenuation,
checkpoint hashes, paired confidence intervals and exact tests. --replay repeats forty
frozen trajectories only, with no weight updates. Analyzer makes new statistics, CSV,
report, caption, vector PDF and 300-dpi PNG; no simulation or training occurs.

The existing Phase 3 full runner documentation describes a full deterministic rerun
into a NEW directory, if separately requested. It is not required for reanalysis.
Current runner differs from the executed source only by the documented metadata-variable
fix. Phases 1/2 use their original commands and archive readers, unchanged.
