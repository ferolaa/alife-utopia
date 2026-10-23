# V2 controlled pilot: technical review

All eight approved runs completed. **This pilot did not demonstrate improved standardized parental care.** Assigned-pup independence was unchanged at every validation checkpoint for every run. No further training, architecture changes, PPO, new crowding mechanism, commit or push followed the null result.

## Executed scope and checks

- 366 tests passed (346 previous + 20 new), including the independently enumerated coupled two-agent gradient test and its cross-credit/time-weight ablations.
- New protocol corrections: cumulative-decision validation, standardized architecture-independent assigned-pup assay, complete parent follow-up, and persisted scheduler state.
- Exact original 500-tick/40-founder/cap-120 training preset; colony credit, gamma 0.99, actor/critic Adam lr 0.001; original 234-parameter MLP only.
- Learned critic: separate 5,057-parameter network; one separate MSE/Adam update per actor update. No normalization, clipping, entropy bonus or search.
- Five fixed validation worlds × energies 10 and 20: ten assigned trials per panel, separate from five natural-colony episodes.
- 40 validation panels: initial and four scheduled thresholds per run; 400 assigned trials and 200 natural episodes, all frozen.
- 40 checksummed checkpoints. All eight final checkpoints restored with exact policy/critic/scheduler state and identical assigned-trial replay. Adam step counters match logged updates.
- All 256 original files, including 178 historical result/artifact files, remain byte-identical. HEAD and remote main/breakdown-mechanisms/dlai-revision-v2 SHAs remain unchanged.
- Source hashes stayed identical throughout collection. Checkpoint schema 2 explicitly distinguishes these corrections. No reserved final-test world was instantiated.

## Realized training budgets

Whole episodes were retained. Prespecified tolerances were ≤20,000 excess decisions per run and ≤20,000 spread between runs. All passed.

| Run | Adult decisions | Overshoot | Actor updates | Critic updates |
|---|---:|---:|---:|---:|
| reinforce-survival-s0 | 202,459 | 2,459 | 31 | 0 |
| reinforce-survival-s1 | 204,846 | 4,846 | 38 | 0 |
| reinforce-offspring-s0 | 200,414 | 414 | 31 | 0 |
| reinforce-offspring-s1 | 204,523 | 4,523 | 38 | 0 |
| reinforce_value-survival-s0 | 203,012 | 3,012 | 31 | 31 |
| reinforce_value-survival-s1 | 206,419 | 6,419 | 38 | 38 |
| reinforce_value-offspring-s0 | 200,319 | 319 | 31 | 31 |
| reinforce_value-offspring-s1 | 206,342 | 6,342 | 38 | 38 |

Total training decisions: **1,628,334**. Realized budget spread: **6,100** (3.05% of target). Maximum overshoot: 6,419 (3.21%).
Validation consumed 1,211,104 additional adult decisions, excluded from training. Eight one-trial checkpoint replays are separately counted in run summaries.

## Exact initialization comparisons

Each initialization is exactly matched across both algorithms and reward conditions. All five scores per run are against the same world/action seed panel; no pooled significance calculation is performed.

| Run | Assigned initial→final | Attendance initial→final | Natural births initial→final | Natural independence initial→final |
|---|---:|---:|---:|---:|
| reinforce-survival-s0 | 9/10 → 9/10 | 59.8% → 57.9% | 23 → 25 | 69.6% → 65.2% |
| reinforce-survival-s1 | 2/10 → 2/10 | 32.7% → 31.7% | 14 → 13 | 14.3% → 38.5% |
| reinforce-offspring-s0 | 9/10 → 9/10 | 59.8% → 57.9% | 23 → 28 | 69.6% → 57.7% |
| reinforce-offspring-s1 | 2/10 → 2/10 | 32.7% → 31.7% | 14 → 11 | 14.3% → 18.2% |
| reinforce_value-survival-s0 | 9/10 → 9/10 | 59.8% → 57.9% | 23 → 21 | 69.6% → 52.4% |
| reinforce_value-survival-s1 | 2/10 → 2/10 | 32.7% → 31.7% | 14 → 11 | 14.3% → 18.2% |
| reinforce_value-offspring-s0 | 9/10 → 9/10 | 59.8% → 57.9% | 23 → 31 | 69.6% → 66.7% |
| reinforce_value-offspring-s1 | 2/10 → 2/10 | 32.7% → 31.7% | 14 → 12 | 14.3% → 25.0% |

For both algorithms/objectives, seed 0 had energy-10 success **4/5→4/5** and energy-20 **5/5→5/5**. Seed 1 had **1/5→1/5** in each energy condition. Seed 0 began near the assay ceiling; initialization differences dwarf the observed training changes. Attendance declined by 1.87 percentage points for seed 0 and 0.96 points for seed 1. The small natural-colony changes are mixed: one seed/condition improves while others decline, with different numbers of eligible births. They do not establish a repeatable parenting gain.

## Every scheduled validation point

Assigned successes are out of ten. Natural success uses eligible births, never the assigned-trial denominator. Exact trial-level records and energy breakdowns are in each run summary.

| Run | Requested threshold | Actual decisions | Assigned successes | Assigned attendance | Natural births | Eligible births | Natural independence |
|---|---:|---:|---:|---:|---:|---:|---:|
| reinforce-survival-s0 | 0 | 0 | 9/10 | 59.8% | 23 | 23 | 69.6% |
| reinforce-survival-s0 | 50000 | 50,357 | 9/10 | 59.8% | 21 | 21 | 71.4% |
| reinforce-survival-s0 | 100000 | 100,351 | 9/10 | 58.4% | 19 | 19 | 57.9% |
| reinforce-survival-s0 | 150000 | 152,541 | 9/10 | 58.1% | 20 | 19 | 57.9% |
| reinforce-survival-s0 | 200000 | 202,459 | 9/10 | 57.9% | 25 | 23 | 65.2% |
| reinforce-survival-s1 | 0 | 0 | 2/10 | 32.7% | 14 | 14 | 14.3% |
| reinforce-survival-s1 | 50000 | 51,073 | 2/10 | 33.4% | 13 | 13 | 7.7% |
| reinforce-survival-s1 | 100000 | 104,257 | 2/10 | 34.7% | 13 | 13 | 30.8% |
| reinforce-survival-s1 | 150000 | 150,414 | 2/10 | 33.5% | 11 | 11 | 0.0% |
| reinforce-survival-s1 | 200000 | 204,846 | 2/10 | 31.7% | 13 | 13 | 38.5% |
| reinforce-offspring-s0 | 0 | 0 | 9/10 | 59.8% | 23 | 23 | 69.6% |
| reinforce-offspring-s0 | 50000 | 50,427 | 9/10 | 59.8% | 21 | 21 | 71.4% |
| reinforce-offspring-s0 | 100000 | 100,788 | 9/10 | 58.2% | 22 | 22 | 54.5% |
| reinforce-offspring-s0 | 150000 | 151,783 | 9/10 | 58.1% | 20 | 20 | 50.0% |
| reinforce-offspring-s0 | 200000 | 200,414 | 9/10 | 57.9% | 28 | 26 | 57.7% |
| reinforce-offspring-s1 | 0 | 0 | 2/10 | 32.7% | 14 | 14 | 14.3% |
| reinforce-offspring-s1 | 50000 | 50,865 | 2/10 | 33.4% | 13 | 13 | 7.7% |
| reinforce-offspring-s1 | 100000 | 103,549 | 2/10 | 34.7% | 10 | 10 | 20.0% |
| reinforce-offspring-s1 | 150000 | 154,874 | 2/10 | 33.5% | 12 | 12 | 8.3% |
| reinforce-offspring-s1 | 200000 | 204,523 | 2/10 | 31.7% | 11 | 11 | 18.2% |
| reinforce_value-survival-s0 | 0 | 0 | 9/10 | 59.8% | 23 | 23 | 69.6% |
| reinforce_value-survival-s0 | 50000 | 50,357 | 9/10 | 59.8% | 21 | 21 | 71.4% |
| reinforce_value-survival-s0 | 100000 | 100,351 | 9/10 | 58.4% | 19 | 19 | 57.9% |
| reinforce_value-survival-s0 | 150000 | 152,876 | 9/10 | 58.1% | 20 | 19 | 63.2% |
| reinforce_value-survival-s0 | 200000 | 203,012 | 9/10 | 57.9% | 21 | 21 | 52.4% |
| reinforce_value-survival-s1 | 0 | 0 | 2/10 | 32.7% | 14 | 14 | 14.3% |
| reinforce_value-survival-s1 | 50000 | 51,349 | 2/10 | 33.4% | 13 | 13 | 7.7% |
| reinforce_value-survival-s1 | 100000 | 104,533 | 2/10 | 34.7% | 12 | 12 | 16.7% |
| reinforce_value-survival-s1 | 150000 | 151,109 | 2/10 | 33.5% | 13 | 13 | 7.7% |
| reinforce_value-survival-s1 | 200000 | 206,419 | 2/10 | 31.7% | 11 | 11 | 18.2% |
| reinforce_value-offspring-s0 | 0 | 0 | 9/10 | 59.8% | 23 | 23 | 69.6% |
| reinforce_value-offspring-s0 | 50000 | 50,357 | 9/10 | 59.8% | 21 | 21 | 71.4% |
| reinforce_value-offspring-s0 | 100000 | 108,613 | 9/10 | 58.4% | 25 | 25 | 60.0% |
| reinforce_value-offspring-s0 | 150000 | 151,035 | 9/10 | 58.1% | 18 | 18 | 55.6% |
| reinforce_value-offspring-s0 | 200000 | 200,319 | 9/10 | 57.9% | 31 | 30 | 66.7% |
| reinforce_value-offspring-s1 | 0 | 0 | 2/10 | 32.7% | 14 | 14 | 14.3% |
| reinforce_value-offspring-s1 | 50000 | 50,865 | 2/10 | 33.4% | 13 | 13 | 7.7% |
| reinforce_value-offspring-s1 | 100000 | 103,983 | 2/10 | 34.7% | 10 | 10 | 20.0% |
| reinforce_value-offspring-s1 | 150000 | 150,015 | 2/10 | 33.5% | 12 | 12 | 25.0% |
| reinforce_value-offspring-s1 | 200000 | 206,342 | 2/10 | 31.7% | 12 | 12 | 25.0% |

## Optimization and failure diagnostics

| Run | Training births / independent offspring | Extinct episodes / total | Failed reproduction / attempts | Entropy first→last | Median actor gradient norm | Critic MSE first→last |
|---|---:|---:|---:|---:|---:|---:|
| reinforce-survival-s0 | 153 / 100 | 6 / 31 | 41,498 / 41,651 | 1.775 → 1.774 | 1322 | N/A |
| reinforce-survival-s1 | 67 / 20 | 36 / 38 | 26,433 / 26,500 | 1.747 → 1.748 | 1177 | N/A |
| reinforce-offspring-s0 | 150 / 101 | 5 / 31 | 41,061 / 41,211 | 1.775 → 1.774 | 1367 | N/A |
| reinforce-offspring-s1 | 75 / 24 | 36 / 38 | 26,396 / 26,471 | 1.747 → 1.748 | 1183 | N/A |
| reinforce_value-survival-s0 | 152 / 99 | 5 / 31 | 41,638 / 41,790 | 1.775 → 1.774 | 1264 | 269.4 → 219.9 |
| reinforce_value-survival-s1 | 67 / 19 | 34 / 38 | 26,683 / 26,750 | 1.747 → 1.748 | 1100 | 297.5 → 230.9 |
| reinforce_value-offspring-s0 | 142 / 95 | 4 / 31 | 41,001 / 41,143 | 1.775 → 1.774 | 1300 | 272.7 → 217.2 |
| reinforce_value-offspring-s1 | 73 / 26 | 35 / 38 | 26,632 / 26,705 | 1.747 → 1.748 | 1120 | 312.0 → 248.8 |

All logged losses, gradients and parameters remained finite. Actor entropy stayed approximately 1.746–1.776 nats, close to the six-action maximum of 1.792; no deterministic action-collapse signal was observed. Actor parameters moved by about 3.9–4.0% in relative L2 norm, so the flat behavioral curves are not caused by absent optimizer steps. Raw gradient norms were large but finite; world-level loss sums intentionally retain the declared objective scaling.

The value baseline slightly lowered some gradient magnitudes and its training MSE ended below its first value, but those are different on-policy episodes. They do not demonstrate estimator variance reduction, held-out critic accuracy or convergence. It produced no assigned-care gain and no consistent natural-colony advantage.

Large fractions of reproduction attempts failed (>99%, primarily opportunity/energy/age/nest constraints; individual causes are not yet logged). There were no cap-rejected births. Seed 1 had 7–9 birth-free training episodes and 34–36 extinct episodes out of 38. These are observed demographic failures, not proof that a deliberate zero-birth strategy was learned. Training episodes ranged from 283 to 500 ticks, so raw episode rewards must be interpreted with duration and exposure.

Offspring conditions produced only 95–101 independence events for seed 0 and 24–26 for seed 1. The explicit offspring bonus contributed roughly 4.6–4.8% and 1.2% of their undiscounted training reward respectively. This quantifies sparse bonus exposure, not its exact share of the gradient. Colony-survival reward also indirectly values descendants.

Four of seed 0's nine assigned successes, and one of seed 1's two, were posthumous at both initialization and final validation. Five parents died per final ten-trial panel for each seed. These retained model rules allow pup independence despite later loss of the parent; success alone is therefore not evidence of improved sustained parental attendance. No increase in this behavior or other clear learned reward exploit was detected. The two-point seed sample and small panel cannot rule out exploitation.

## Remaining limitations and next recommendation

1. Do not promote these models or start the 120-model campaign/PPO. The required actual-environment parenting improvement has not been demonstrated.
2. Independently review the uncommitted corrections and frozen evidence, especially the joint colony objective, assigned-pup follow-up, sparse reproductive opportunity and the finite-horizon/time discount definition.
3. The pilot has only 31–38 actor updates per run; decision exposure is not optimizer exposure. Only two initializations and five worlds were used. The near-ceiling seed-0 baseline makes this panel insensitive to some improvements.
4. A next separately approved diagnostic should prespecify how to distinguish lack of reward signal, inadequate value fitting and assay insensitivity. Instrument reproduction failure reasons, critic calibration/explained variance and gradient signal-to-noise before choosing a new training budget or reward change. Do not tune repeatedly until a positive score appears.
5. No new arithmetic blocker was exposed in the supported CPU completed-episode path. The old episode-frequency API remains for compatibility but was disabled in all eight runs. Checkpoint schema/source mismatch is deliberately rejected rather than silently migrated.
6. Full source, protocol, initialization panels, exact world/action seeds, JSONL episode logs, all trial records and intermediate checkpoints are preserved. No final-test panel has been opened.

## Curves and evidence

- `/workspace/dlai-pilot-v2-001/assigned-independence.png`: all four conditions, both seeds, actual decision axes.
- `assigned-attendance.png`, `assigned-by-energy.png`, `natural-births.png`: separate descriptive outcomes.
- Each run has `learning-curves-readable.png` with assigned/natural outcomes, reward/discounted return, entropy, gradients, critic MSE and episode duration.
- `validation-points.csv`: all 40 points. `budgets.csv`, `behavioral-diagnostics.csv/json`, `parameter-changes.json`: exact numeric evidence.
- `FINAL_VERIFICATION.json`, `historical-before/after.json`, `preflight.json`, source/review snapshots and artifact SHA-256 manifest support independent inspection.

## Code inventory

Modified: `src/v2/__init__.py`, `src/v2/environment.py`, `src/v2/experiment.py`. Added: `src/v2/care.py`, `src/v2/pilot.py`, `experiments/pilot_v2.py`, `tests/test_v2_pilot.py`, `docs/V2_PILOT_PROTOCOL.md`, and this technical review. No original source/experiment/result file changed. All changes remain uncommitted and unpushed.

Phase stops here for user review. No statistical significance is claimed from two training seeds.
