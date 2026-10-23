# Supplied-care implementation review — 2026-10-23

Implementation and bounded validation are complete on `dlai-revision-v2`, based on `66fe988f5774c744f245db2270278fb69747896a`. **No scientific pilot, final-test evaluation, merge, PPO or full-colony/crowding campaign was performed.** The engineering implementation is authorized for a separate verified commit on this branch; scientific execution remains unapproved. This review does not replace the eight-run negative pilot or its diagnosis.

## What changed in this task

New files:

| File | Purpose |
| --- | --- |
| `src/v2/care_task.py` | Versioned real-family initialization, 100-tick collection, feeding/outcome auditing, sealed splits, matched frozen panels and explicit fixed controls |
| `src/v2/care_experiment.py` | Existing V2 learner adapter: four complete worlds/update, threshold validation, safe hashed checkpoints, full batch audit records and fixed-budget single-cell API |
| `experiments/care_feasibility_v2.py` | Bounded review CLI: baselines and optional six one-batch compatibility checks; no campaign launch flag |
| `tests/test_v2_care_task.py` | 48 task-specific regression/behavior/mathematical test cases |
| `docs/V2_CARE_FEASIBILITY_PROTOCOL.md` | Complete proposed eight-cell protocol, interpretation and resource estimate |
| `docs/V2_CARE_FEASIBILITY_REVIEW.md` | This implementation review |
| `docs/V2_NULL_PRESERVATION.json` | Reviewed negative-result preservation references and checksums |
| `docs/V2_CARE_CHANGE_INVENTORY.csv` | Exact review inventory, distinguishing inherited WIP from this task's additions |

The initial implementation added the eight files above without editing the pre-existing project files. Before that stage, the checkout already contained the prior pilot's three modified files (`src/v2/__init__.py`, `src/v2/environment.py`, `src/v2/experiment.py`) and six untracked files (`src/v2/care.py`, `src/v2/pilot.py`, `experiments/pilot_v2.py`, `tests/test_v2_pilot.py`, `docs/V2_PILOT_PROTOCOL.md`, `docs/V2_PILOT_REVIEW.md`). They remain byte-identical to the pre-task preservation snapshot. The complete diff against the committed base therefore includes 17 paths: those nine inherited paths plus these eight additions. No generated datasets, checkpoints, temporary files or new dependency pins were placed in the checkout.

The original 256 files at `ce129bef00de069d171c99b797f142f0711fb626`, including historical results/checkpoints/archives, remain byte-identical. The committed V2 base is additionally preserved as a verified Git bundle with complete ancestry. Current working-source hashes intentionally differ from historical checkpoint source hashes: replay historical models with their archived source, not by bypassing source checks.

## Task and behavioral evidence

The new task supplies a real pup rather than waiting for rare successful reproduction. All adult actions come from the evaluated policy. The original body mechanics and rewards are reused, with an explicit task intervention: one parent/pup, 18 initial conditions, smaller varied layouts, one food patch, no additional reproduction, and 100-tick finite horizon. The pup matures naturally at tick 60; both agents remain under neural control afterward. Pup death does not censor parental follow-up, and parent death does not suppress legitimate later independence.

The final validation panel is 4 predeclared geometry seeds × 18 conditions = 72 supplied families. The geometry seeds are shared across all policies. Counts below are trials, with attendance pooled over actual observed dependent ticks. They are descriptive controls, not trained models or 72 independent scientific replicates.

| Frozen policy | Independent / 72 | Attendance (attended / dependent ticks) | Living-parent independence / 72 | Parent alive at tick 100 / 72 | Posthumous successes | Parent feeding events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untrained neural seed 0 | 36 (50.0%) | 56.18% (2123/3779) | 24 | 8 | 12 | 40 |
| Untrained neural seed 1 | 16 (22.2%) | 38.75% (1303/3363) | 11 | 13 | 5 | 50 |
| Original heuristic | 48 (66.7%) | 71.23% (2815/3952) | 44 | 30 | 4 | 143 |
| Fixed care | 48 (66.7%) | 74.55% (2952/3960) | 24 | 0 | 24 | 0 |
| Abandonment | 0 (0%) | 5.45% (108/1980) | 0 | 28 | 0 | 116 |
| Care + foraging control | 60 (83.3%) | 82.80% (3274/3954) | 60 | 56 | 0 | 701 |
| Uniform actions | 26 (36.1%) | 47.72% (1713/3590) | 16 | 9 | 10 | 48 |

For energy 6 specifically (24 trials/policy), fixed care achieves **0/24**, the heuristic **10/24**, and care + foraging **18/24** independence. Initial neural seeds achieve 3/24 each. Thus proximity alone cannot solve the hardest energy stratum, and an observation-only controller can demonstrably forage and care successfully. The heuristic and care/foraging control both fail some conditions. We make no claim that all conditions are solvable by the original memoryless MLP, nor that the fixed control is an oracle. Food visibility at distance 8 and no history/time input are intentional task difficulties to review.

The original two-energy development panel (10/20) allowed fixed-care independence in every condition, despite parental starvation. That weakness was found and corrected before scientific training; its preliminary frozen outputs are retained. The new panel remains a developed validation panel, not a pristine final test. Success after parental death is a real model outcome, not a shortcut to remove: the review must consider it alongside feeding and living-parent outcomes. Energy-12 fixed care can still achieve posthumous independence, transparently exposing the difference between offspring success and joint survival.

## Fixed-care survival audit before engineering commit

The apparent conflict is real biological behavior under the defined simulator, not inconsistent denominators or censored follow-up. Fixed care moves toward its pup, then stays; it never feeds. Every parent dies of starvation before tick 100. Each energy stratum contains 24 supplied families:

| Starting energy | Parent death tick (all 24) | Independent pups | Parent alive at independence | Posthumous independence | Parent alive at tick 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 22 | 0/24 | 0/24 | 0/24 | 0/24 |
| 12 | 44 | 24/24 | 0/24 | 24/24 | 0/24 |
| 20 | 73 | 24/24 | 24/24 | 0/24 | 0/24 |

For energy 6, pups die of neglect at tick 47. For energy 12, the pup's neglect tolerance bridges the gap from parental death at tick 44 to natural independence at tick 60. For energy 20, independence occurs at tick 60 with a living parent, followed by parental starvation at tick 73. Independent pups survive the remaining follow-up in these fixed-care trials. Thus 48/72 independence splits exactly into 24 living-parent and 24 posthumous successes; 0/72 parents survive to tick 100. Attendance is 2952/3960 dependent exposure ticks. These metrics answer different questions and are all retained. Independence alone does not establish successful joint care/foraging or sustainable parenting.

The final review also replaces `assert world.claim_nest(...)` with an explicit checked claim: optimized Python must not skip the side effect that establishes genuine nest ownership. All 18 conditions were initialized successfully under `python -O`; standard baseline outcomes remain exactly unchanged. No other task mechanics or reward coefficients were changed in this engineering review.

## Validation and reward checks

Final engineering verification: **414 passed in 14.84 seconds**, using the existing pinned virtual environment and two CPU threads. This comprises the previous 366 cases plus 48 new cases. Command:

```bash
env XDG_CACHE_HOME=/workspace/alife-onboarding/cache \
    MPLCONFIGDIR=/workspace/alife-onboarding/matplotlib \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    .venv/bin/python -m pytest -q --tb=short
```

New tests cover real family/nest ownership and all 18 initial conditions; unchanged body parameters; original senses/actions; natural maturation and the exact once-only tick-59 bonus; death on maturation without reward; posthumous attribution; continued follow-up after pup death; extinction/absorbing follow-up; external truncation rejection; no reproduction; RNG/split isolation; matched objectives; frozen validation; feeding instrumentation neutrality; decision scheduler; all six algorithm/architecture combinations; exclusive checkpoint writes, corrupted checksum rejection and exact optimizer continuation.

The mathematical checks use actual Agent/World outcomes: a near-neglect pup survives or dies depending on one initial parental action. Exact expected discounted return is differentiated over both possible actions and compared with the colony policy-gradient estimator, including a constant detached baseline. A second check independently builds the full four-world actor objective and compares every actor gradient tensor with `Learner.update`, for both algorithms. Earlier coupled two-agent enumeration tests remain in the suite and detect omitted cross-agent credit or missing outer gamma factors. These are mathematical correctness checks, not evidence that a neural policy has learned care.

The existing learner is reused unchanged: raw real-clock Monte Carlo returns; gamma^t outer weighting; world-averaged score sums; advantages detached from the critic; critic predictions obtained before fitting; independent actor/critic Adam updates. The new adapter never truncates an episode for budgeting. Survival-only and offspring collection at the same weights/world seed produce identical transitions and survival events; only the additional offspring events differ.

The review CLI successfully performed **one four-world update per algorithm/architecture** (six combinations), checked finite diagnostics and safely restored each new checkpoint. Across the six successful final smoke cells there were 1,616 adult decisions; these are technical checks, not the eight-cell pilot. The tests also perform bounded optimizer checks. An earlier checkpoint smoke failed because NumPy observation scalars were serialized into the audit metadata; this was corrected by primitive-only metadata serialization, without permissive pickle loading. A final family audit also sets the parent's cumulative child count to one for the supplied prior birth, while keeping new in-task births at zero. That failed attempt and all earlier review outputs are retained. No saved historical checkpoint was modified.

## Preservation and provenance of the negative result

The negative pilot remains eight complete runs totaling 1,628,334 training decisions, with all original configurations, JSONL logs, 40 checkpoints/hashes, curves and summaries intact. Its initial-to-final assigned-pup independence results were unchanged in every cell; expanded frozen diagnostics did not rescue a positive result. Keep that conclusion and all failed runs.

Directly recorded historical quantities: **272,221 reproduction attempts, 271,342 failures (99.677%), 879 births, zero cap rejections**. The original logs did not record individual failure causes. Exact frozen replay reconstructs **32 of 276** historical training episodes, covering 32,999 attempts and 32,875 failures: first failing gate energy 26,017, age 1,600, nest 5,258, plus 124 births. These are causes observed during validated reconstruction, not cause labels logged during the original training. The other **238,467 failed attempts** have unknown causes; 244 missing pre-update policies prevent exact reconstruction. Overlapping gate flags in the diagnosis must not be summed as mutually exclusive causes.

The reviewed preservation archive has **507 members**, with every member decompressed and SHA-256 checked against the exact inclusion inventory. Its 506 data members comprise 143 pilot-directory files (142 manifest-listed + manifest), 86 diagnostic-directory files (84 manifest-listed + manifest + retained diagnostic Python cache), the two original ZIP files, original pilot console log, 273 pre-task checkout files and the pre-change baseline inventory. The harmless historical cache is included for losslessness; no code depends on it. Nothing unsuccessful was filtered out. Existing original archive hashes and every original artifact-manifest entry were independently rechecked.

See `docs/V2_NULL_PRESERVATION.json` for paths and checksums. The separate verified Git bundle preserves the V2 base and complete V1 ancestry. The final review package includes this preservation material, exact inventories, all review stages, current source/tests/docs and final test/smoke results. All generated material lives under `/workspace/dlai-care-review-v3-001`, outside the checkout.

## Frozen identities and matching before engineering commit

All 504 frozen-policy records (seven policies × 72 trials) reproduced the previously reviewed records exactly after canonical JSON serialization, including world/action seeds, initial conditions, outcomes, counters and policy hashes. A second complete fixed-care panel replayed exactly. All 72 world/condition pairs matched across policies. Weight hashes, mode and ambient Python/NumPy/Torch RNGs were unchanged by evaluation; no evaluation optimizer update occurred. Full state SHA-256 identities:

| Policy | State SHA-256 |
| --- | --- |
| untrained-s0 | `5d918e2d415a8c42e9610fb24bf77ba3c279d9738b1899c7c2c9445509547c01` |
| untrained-s1 | `d4dbaba5532287cd093e4d91444292704b4683a8712215ecef0affbf456feac3` |
| heuristic | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| fixed-care | `163de6ac155f6320abdf80ab31b84605aa0da66ae87eb5f37928f8b007ee9f44` |
| abandonment | `0ed377d99402bb32808d2be1c701adf5cc1d44c12c94ebf2f178bd64d01e969e` |
| forage-care | `2d11735fdb790dae88649e97072cad7850d47aceda2b3e018808f75f985b6749` |
| uniform | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The heuristic and uniform policy have empty state dictionaries and therefore the same empty-state digest. They are distinguished by their explicit class (`HandDesigned` versus `Uniform`) and implementation source, not treated as the same model or cached under that digest. Fixed-control identities additionally contain a behavior buffer. Neural identities match the reviewed seed-0/seed-1 original MLPs.

The complete 414-test run includes exact actor/critic optimizer continuation and deterministic next-batch replay for both algorithms and all three architectures, plus checkpoint checksum and overwrite protection. The scientific fixed-budget runner was exercised only with synthetic counters in its stop-rule test. No eight-cell pilot or final panel was executed.

All 256 original V1 files, including 178 historical result files, remain byte-identical to the preserved original commit. Every manifest entry in the old null pilot (142) and diagnosis (84), both original ZIP hashes, all 507 members of the lossless preservation archive, the original review-package hash and preserved foundation bundle were independently rechecked. Those outputs remain outside the commit. Full verification records are retained under `/workspace/dlai-care-commit-review-001`.

Repository-local author and committer are both `Mariam Latsabidze <65814453+mariamlats@users.noreply.github.com>`. The proposed commit contains only the 17 source/test/documentation paths in `V2_CARE_CHANGE_INVENTORY.csv` (3 modified foundation paths and 14 additions). No checkpoint, per-trial dataset, archive, cache, credential or dependency change is included. The preserved historical pilot runner remains pinned to its original reviewed foundation; it is not the launcher for the new care pilot.

## Proposed pilot and remaining review decisions

The complete protocol is in `V2_CARE_FEASIBILITY_PROTOCOL.md`: 2 algorithms × 2 objectives × seeds 0/1; original MLP; 20k adult decisions/cell; four complete worlds/update; initialization and ~5k-interval frozen validation; fixed reward coefficients/gamma; equal zero tuning budgets; all batch boundaries saved. Maximum whole-batch excess/spread is 563 decisions; report actual episodes/updates and interactions. Typical early-smoke projection is ~65 updates/cell, but no learning pilot was run to establish that projection. Reserve 4–10 CPU minutes and approximately 1–4 GB for the eventual eight-cell campaign.

There is **no known failing correctness check** after the fixes. Remaining pre-training blockers are independent code/protocol review, explicit scientific execution approval, and a small authorized campaign launcher/preflight around the tested single-cell budget API. That launcher must enforce the declared eight-cell matrix, original-archive checks and aggregate budget spread; the review CLI intentionally cannot launch it. No approval is inferred from completing this implementation.

Scientific limitations remain: only two training replicates; four reused validation geometry seeds; no memory/time input; the value baseline may be poorly calibrated; a 20k budget probes early feasibility, not convergence; independence can be obtained by parental sacrifice; changed tasks/batching prevent attributing differences from the old null pilot to the learning algorithm alone. Report outcomes and heterogeneous seed behavior at the fixed budget without arbitrary every-seed success gates, significance claims or automatic budget growth. Full-colony generalization remains untested.
