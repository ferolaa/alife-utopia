# Full Phase 3: frozen-policy sensitivity to programmed cue attenuation

The approved matrix was run once: ten paired training seeds × two frozen objectives ×
twenty seed triplets × normal/disrupted childhood × assignment OFF/ON = 1,600 complete
trajectories. Childhood seeds are 410000–410019, bridge seeds 420000–420019 and adult
seeds 400000–400019. No dose, seed, network, outcome or scientific parameter changed.

## Result

Each cell contains 200 evaluations. All three zero-impairment controls are identical:
survival-only 22/200 (11%); offspring-reward 114/200 (57%). Disrupted/ON, with impairment
0.30, produces 20/200 (10%) and 84/200 (42%). All 1,600 subjects survive childhood and
transfer; all 1,600 parents survive the full assessment. All unsuccessful assigned pups
die of neglect; no starvation, old-age death or posthumous independence occurs.

The difference-in-differences is (disrupted/ON − normal/ON) −
(disrupted/OFF − normal/OFF). Negative values indicate reduced independence.

| Contrast | Mean pp | 95% paired bootstrap interval | Exact p | Holm p |
|---|---:|---|---:|---:|
| Survival-only | −1.0 | −4.5, +1.5 | 1.000000 | 1.000000 |
| Offspring-reward | −15.0 | −21.0, −9.5 | 0.001953 | 0.005859 |
| Offspring minus survival | −14.0 | −20.5, −8.5 | 0.001953 | 0.005859 |

Inference uses ten training-seed pairs, not 1,600 independent observations. Bootstrap:
100,000 paired resamples, seed 20261014; intervals are unadjusted percentile intervals
conditional on the fixed twenty-world panel. Exact two-sided tests enumerate all
1,024 sign patterns, with the sign-exchangeability/symmetry assumption under the null.
Holm adjustment covers the three planned contrasts only. The first two training pairs
and first three worlds appeared in the pilot; the protocol was not changed afterward.

Offspring-policy effects are negative in all ten seeds: −25, −5, −35, −20, −5, −5,
−20, −15, −10, −10 pp. Survival effects: 0, +5, 0, −15, 0, 0, 0, 0, 0, 0 pp.
Matched outcome flips: survival loses four successes and gains two; offspring loses
32 and gains two. Individual improvements are therefore present despite the average loss.

Attendance is 22.29% versus 21.70% for survival controls/impaired, and 54.59% versus
44.62% for offspring. Parent meals are 891/979 and 352/515 respectively per 200 trials.
Attendance uses observed care ticks, ending at death/maturation; raw attendance and
attended-per-planned-tick measures are also saved. Mortality always has full parental
follow-up. The survival-policy baseline is low: its smaller absolute loss must not be
called greater intrinsic resilience. Offspring policies still outperform survival
policies after attenuation (42% versus 10%).

## Report-ready interpretation

After a fixed sublethal childhood care-disruption exposure, the model's programmed
persistent attenuation of own-pup cues reduced offspring-reward policy independence
success from 57% to 42% in uncrowded adult assessments. The reduction occurred in all
ten training seeds and was supported by the causal assignment ablation. Survival-only
policies showed a small mixed change, with a low baseline limiting the comparison.
This demonstrates downstream sensitivity of frozen policies to an engineered encoding
shift. It does not discover persistence, neural damage, biological trauma, inherited
learning, genuine recovery or a Calhoun population-collapse mechanism.

This result warrants one small effect panel or paragraph in the three-page report,
with full cell rates, seed profiles, controls and diagnostic details supplementary.
Phases 1 and 2 remain the principal learning and crowding results. The two-panel Phase 3
figure is supplied as vector PDF and 300-dpi PNG with a precise caption.

## Execution, verification and metadata correction

Measured experiment runtime: **117.485 seconds**. Preflight: all 265 tests passed.
After adding two metadata regressions, all **267 tests pass**. All thirteen technical
gates pass. Independent verification covers 1,602 raw files, 400 identical control
contexts, 96,000 same-state sensory audits, all twenty saved checkpoints and 40 exact
frozen trajectory replays. All 48 pilot trajectories match corresponding full records.
All core/Phase 1/2 and pilot outputs remain unchanged; no commit or push occurred.

A packaging defect was discovered after the run: the local expected-count variable
was reused for a checkpoint hash. Thus `expected_trajectories` in the original raw
manifest and summary is a hash string, while actual `trajectories` is correctly 1,600.
It did not affect matrix iteration, physics, outcomes, weights, RNG or inference.
The complete filename/key matrix independently proves the actual count.

Original outputs have NOT been edited or rerun. The runner now uses a distinct
checkpoint-hash variable, and tests check both 48/1,600 counts. The explicit sidecar
`METADATA_CORRECTION.json` records correct count, original value and source hashes.
`raw_run_runner_source.txt` preserves the exact executed full version; the pilot source
is also preserved. The independent verifier requires and validates the sidecar for
these original outputs. Preserve all three files alongside the dataset before publication
or Git archival; do not silently repair or discard the historical manifest.

## Reproduction commands

Run from the repository root with the pinned CPU environment. Every target must be new.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python experiments/verify_developmental_care.py --replay
.venv/bin/python experiments/analyze_developmental_care.py --out results/developmental_care_reanalysis_new
.venv/bin/python experiments/developmental_care.py --full --out results/developmental_care_reproduction_new
```

The last command runs the full approved matrix again and is documentation, not an
instruction to rerun during review. Future runs produce correctly typed metadata.
The original source snapshots are provenance, not directly executable from their
data directory; use the runner under `experiments/`. Figure/statistic generation is
portable and depends on saved raw records; analysis never retrains a policy.

## Preservation review

Raw data: `results/developmental_care_v1/` (1,602 files, about 313.73 MB).
Analysis/figure/verification/correction/source snapshots:
`results/developmental_care_v1_analysis/`.
New scripts: `analyze_developmental_care.py`, `verify_developmental_care.py`.
Updated Phase 3-only runner/tests: `developmental_care.py`, `test_developmental_care.py`.
No core mechanics or earlier experiment files changed. Lossless archival is recommended
before staging the large raw dataset; none of these Phase 3 files has been committed.
