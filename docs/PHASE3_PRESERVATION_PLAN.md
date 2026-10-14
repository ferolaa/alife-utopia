# Phase 3 atomic preservation review — no commit or push yet

Base: 79b7a29df0ed53c6ef818f93ee2708354274fb83, breakdown-mechanisms.
Suggested single commit: `Preserve verified developmental-care experiment`.
The authoritative explicit inclusion list is `docs/PHASE3_INCLUSION.json`.
Per-file sizes and SHA-256 hashes are in `docs/PHASE3_INVENTORY.csv`.

Include the four Phase 3 scripts, tests, pilot/full protocol documents, this plan,
portable reproduction instructions, original pilot/full manifest and summary,
all existing Phase 3 analysis/provenance/correction files, and both lossless archives
with per-record indices. Include the new preservation verification receipt.
All dependencies already exist in requirements-reproduction.txt; packaging uses only
Python's standard library. No checkpoint duplication is needed: all twenty required
checkpoints and their evaluation hashes are already committed in final_reward_v1.

Exclude all 1,648 uncompressed trajectory JSON files (1,600 full + 48 pilot): their
exact bytes are archived. Keep the local originals, do not delete or rewrite them.
Exclude seven superseded Phase 2 presentation files, caches, virtual environments,
logs, restoration/reanalysis copies and scratch files under /workspace/work.
No existing tracked files or core mechanics are changed. Legacy results stay tracked.

The original expected_trajectories metadata defect remains in both original full
outputs. METADATA_CORRECTION.json and original executed source snapshots MUST accompany
these files. Verification uses the independently enumerated 1,600-key matrix and
validates the explicit sidecar. Never silently edit the historical outputs.

Verification performed: 267 tests; 1,600 complete records, 400 exact zero-impairment
control contexts, 96,000 sensory checks, twenty frozen checkpoints and forty exact
replays. Full and pilot archives restore all 1,652 files byte-identically. Portable
reanalysis reproduces analysis.json, PNG, CSV, caption and report text byte-identically.
PDF is regenerated with the same plotted values; generated PDF dates are not a
scientific reproducibility target. A local fresh clone plus ONLY proposed additions
was tested using the existing pinned environment. Installing dependencies on a newly
provisioned machine remains untested; do not substitute unavailable pins silently.

Figure review: accurate percentage/percentage-point units, fixed 0–100% rate axis,
paired training-seed intervals, explicit control labels and ten individual seed effects.
For three pages, use panel B and a short supporting paragraph; keep panel A supplementary.
No data/presentation fixes were necessary. The result measures frozen-network sensitivity
to a programmed persistent 30% reduction in three own-pup inputs, not discovered
persistence, trauma, inherited learning, neural damage or recovery. The survival-only
11% baseline limits comparisons of apparent resilience. All subjects/parents survive;
failed assigned pups die of neglect. Intervals are unadjusted and conditional on the
fixed twenty-world panel. Sign-flip inference assumes null sign exchangeability.

After separate approval: stage ONLY paths in the inclusion JSON (never `git add .`),
check staged names/sizes/diff --check, rerun tests/archive verification, confirm identity,
commit atomically and push to origin/breakdown-mechanisms without force or PR. Confirm
remote SHA and artifacts. Do not modify or merge main.

Identity confirmed locally: Mariam Latsabidze
<65814453+mariamlats@users.noreply.github.com>. This persists in this checkout;
fresh clones need their own local identity configuration.
