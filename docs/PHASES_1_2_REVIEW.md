# Final scientific review of Phases 1 and 2

Both experiments use ten paired training seeds, CPU frozen shared policies, and
matched evaluation worlds. Training-world sets (seed × 1000 + iteration) are
disjoint across training seeds: 1,200 unique worlds. Initial weights match within
objective comparisons; all ten initial weight sets and all twenty trained weight
sets are distinct. Evaluation seeds/action RNGs are deliberately shared for matching.
Inference treats training pairs as replicates and conditions on the fixed world panel.

## Phase 1: learning parental behavior

The prespecified primary endpoint is assigned-pup independence, not directional pup
seeking. Each model has 40 trials: 20 worlds × energies 10/20. There are 400 trials
per objective, but only ten independent training pairs.

Survival-only: 39/400 (9.75%); survival-plus-offspring: 203/400 (50.75%). Mean paired
advantage: **41.0 percentage points**, 95% paired bootstrap interval **33.25–49.0**;
exact two-sided sign-flip **p = 0.001953125**. All ten pair differences are positive.
The bootstrap uses 10,000 paired resamples, NumPy RNG seed 20261014.

Natural-colony recruitment is descriptive and conditional on being born. Survival:
4,333/6,023 eligible offspring reached independence; offspring objective:
14,305/14,723. Late births without 60 ticks of follow-up are excluded, not failures.
Final mean populations were 16.095 and 54.645; extinction counts 22/200 and 5/200.
The offspring group reached the 120-agent cap in 19/200 worlds; survival reached it
in none. Colony outcomes therefore do not establish unconstrained population growth.

Natural deaths: survival 5,643 starvation / 3,802 old age / 1,701 neglect; offspring
8,398 / 4,005 / 424. Raw starvation totals are not mortality risks: population and
adult exposure differ, and replenishment does not guarantee individual food access.

Five of 39 survival-policy successes and 57 of 203 offspring-policy successes occurred
after parent death. As an explicitly post-hoc descriptive check, independence with a
living parent was 34/400 versus 146/400; it is not a replacement primary endpoint.
Parent follow-up ends when the pup dies: 357 survival-group and 174 offspring-group
parents were alive at an early stop. The observed 9 versus 80 parent deaths cannot
be compared as equal-horizon 60-tick mortality risks.

## Phase 2: controlled crowding and care disruption

The matrix has 3,200 unique trials: 10 training seeds × 20 worlds × 2 objectives ×
2 energies × 2 background counts × 2 disruption flags. Each energy-specific cell
has 200 trials; each pooled-energy cell has 400. Every trial follows parents for
60 ticks. Care is observed only while a pup is dependent (maximum 59 care ticks).

| Background bodies | Disruption | Survival E10 | Offspring E10 | Survival E20 | Offspring E20 |
|---:|---|---:|---:|---:|---:|
| 0 | OFF | 19/200 | 106/200 | 19/200 | 123/200 |
| 0 | ON | 18/200 | 102/200 | 18/200 | 123/200 |
| 8 | OFF | 19/200 | 107/200 | 21/200 | 123/200 |
| 8 | ON | 0/200 | 3/200 | 0/200 | 10/200 |

Pooled high/ON offspring advantage: **3.25 points** (95% interval **1.50–5.25**),
exact p=0.015625, Holm p=0.046875. The policy × crowding × disruption contrast is
[(offspring−survival) high/ON − high/OFF] − [(offspring−survival) low/ON − low/OFF]:
**−43.75 points**, interval **−53.0 to −35.0**, exact p=0.001953125, Holm p=0.017578125.

High-crowding disruption penalties: survival −10.0 points (−14.0 to −6.5), offspring
−54.25 (−61.5 to −47.5). Larger absolute offspring-policy loss does not establish
intrinsically poorer resilience: survival starts near 10% and reaches a complete floor.
All ten survival seeds and three offspring seeds have zero high/ON success.

Bootstrap: 100,000 paired training-seed resamples, seed 20261014. Sign-flip tests use
all 1,024 patterns. Holm correction is within each nine-contrast family; energy-specific
analyses are secondary. Intervals are unadjusted, approximate with n=10, and not a
joint guarantee across contrasts. The nine-contrast correction is a reporting choice,
not a claim of preregistration. Low-background offspring disruption has an interval
excluding zero but an exact p=0.125; do not call it significant based on the interval alone.

Offspring-policy attendance changes from 49.45% to 8.41% at high crowding; observed
crowd-blocked ticks rise from 0 to 5,119. Attendance denominators end at pup death or
independence and vary across conditions. They are not common-horizon mortality measures.
All E20 parents survive; independence nevertheless falls 61.5% to 5%. The effect cannot
be explained solely by parental starvation. All unsuccessful pups die of neglect;
all parental deaths are starvation in E10; no old-age deaths occur.

Offspring-policy successes after parent death: 79/229 low/OFF, 75/225 low/ON,
79/230 high/OFF and 2/13 high/ON. Posthumous independence is allowed by neglect
tolerance, rather than proof of ongoing care. These events are outcome observations,
distinct from the training algorithm's posthumous reward attribution limitation.

## Defensible conclusion and limitations

Adding offspring reward improves learned parenting outcomes under the tested training
and evaluation algorithm. High stationary crowding alone leaves independence almost
unchanged, but the programmed care-distance disruption largely removes the advantage.
Frozen pretrained policies do not robustly preserve recruitment under that imposed stress.
This is not a colony-collapse experiment or biological replication of Calhoun.

The disruption law is supplied by the modeller, not discovered from behavior. Background
bodies are stationary occupancy proxies, not competing animals. They affect normal policy
crowding inputs; off/on contrasts within density isolate the care intervention, while
cross-density comparisons include responses to perceived crowding. Crowding energy
penalties, intrusion, developmental damage and further reproduction are disabled in
this assay. Parent-inclusive tick-start counts coexist with post-movement attendance,
allowing some blocking even with zero background bodies; that timing was preserved.

Care is proximity-based and does not model milk transfer, mating, hierarchy, recognition
or biological development. No untrained-policy comparator or learned neural semantic
representation claim is established. Shared, feed-forward policies do not learn during
evaluation. Episodic horizon truncation and posthumous offspring-reward credit remain
training limitations; discarded dead-individual trajectories and continuous-window
credit limitations belong to other training modes, not the frozen evaluation loop.

The first three Phase 2 worlds were pilot worlds. No parameters changed after the pilot;
the panel is held out from training, not wholly unseen after protocol development.
Numerical replay has been verified in the recorded CPU environment, not all platforms.

## Figure review

Use the new `results/report_figures_v1` figures and supplied captions. Policy colors and
percentage units are consistent; penalties use percentage points. Error bars resample
training seeds. Background counts are labelled 0/8, not biological population density.
Care plots explicitly use observed care ticks; the supplemental penalty plot warns of floors.

Legacy `results/density.png` is exploratory, not evidence that the current colonies never
starve or that nesting is irrelevant. `results/representation.png` overstates semantic
representation evidence; its ablations do not show a representation is simply present
or absent. `results/rl_comparison.png` measures directional seeking and training births,
not successful care. These legacy files are preserved unchanged and excluded from the
final three-page scientific narrative. Earlier Phase 2 presentation files are retained
locally; the proposed commit uses the reviewed final figures instead of duplicate plots.
