# Atomic neural-policy preservation proposal — no staging, commit or push yet

Branch breakdown-mechanisms, base 3c6a601a7517c20780b0725d77d3663beba2bc5c.
Suggested commit message: `Preserve verified neural-policy evaluation`.
Stage ONLY the paths in docs/NEURAL_INCLUSION.json after separate approval.
Exact per-file sizes and SHA-256 hashes: docs/NEURAL_INVENTORY.csv. Inventory excludes
its own self-referential hash; the inclusion list includes the CSV itself.

Include evaluation/analysis/verifier/response-verifier/archive scripts, regression
tests, pilot/protocol/full reports, these preservation/reproduction instructions,
full and pilot original manifest/summary/integrity metadata, the full lossless ZIP
and per-member index, all original full analysis/figure/provenance files, and a new
preservation verification receipt. No checkpoint copies are needed: the required
20 trained checkpoints and saved starting hashes are already committed in Phase 1.
The archive uses the existing committed flat-JSON verifier/restorer; this dependency
is available from the base commit and remains unchanged.

Exclude full records.json (93MBdataset), pilot records.json (exact60-trial subset),
seven superseded Phase 2 presentation files, earlier uncompressed Phase 2/3 duplicates,
restoration/reanalysis/replay outputs under /workspace/work, caches, venvs and logs.
Retain all local originals unchanged. Archived bytes restore all four full files;
pilot record bytes reconstruct exactly from the full subset and committed pilot
metadata/integrity hash. Do not duplicate the raw pilot trajectories in Git.

No earlier tracked files, core dynamics, networks, saved Phases 1–3 or legacy results
change. No experiment was rerun or trained during packaging; frozen replay and
saved-response recalculation are verification only. Original neural outputs and
pilot remain unchanged. Full manifests/summary have no count defect. Keep historical
Phase 3 metadata correction and source snapshots exactly as committed.

Verification: 286 tests; 2,000 unique keys; 800 exact historical normal trained evaluations;
71,024 input audits;10 distinct exact starting networks shared across objectives;
20 frozen checkpoints;100 additional exact replays. All four paired seed contrasts,
100,000 resample intervals,1,024-pattern exact tests and Holm family recomputed
independently. Every stored probability response and KL was also recalculated from
frozen weights, with 2e-6 absolute / 1e-5 relative float32 tolerance for batch-rounding.
All 1,906 prior protected artifact hashes and original neural hashes are unchanged.

Figure review: vector PDF and 300-dpi PNG are readable at full-width report placement,
0–100% axes, consistent colors, ten seed dots/paired lines, mean diamonds and unadjusted
seed-bootstrap intervals. Use this compact two-panel figure to support objective-
specific learned caregiving and own-pup information reliance. Do not infer hidden
representations, increased depth necessity or continuous care from independence.
Posthumous success, early-stop unequal follow-up, masking distribution shift and
ten-seed/fixed-world uncertainty remain explicit. Survival training's Holm comparison
is nonsignificant despite its unadjusted interval excluding zero. No figure/data fixes
were required. Font-cache warnings do not affect the reviewed rendered artifacts.

After approval: verify explicit staged file set and hashes, run diff --check and
full tests/archive/reanalysis checks, confirm author AND committer, commit atomically,
push ONLY origin/breakdown-mechanisms, then verify remote SHA/artifacts in fresh clone.
No force push, PR, merge, main changes or additional scientific experiments.

Configured identity: Mariam Latsabidze
<65814453+mariamlats@users.noreply.github.com>. Repository-local config persists in
this checkout; fresh clones require their own config. Dependency installation on a
new machine remains untested; the existing pinned CPU environment was used for the
clean-clone file-set validation. No secrets were identified in the proposed files.
