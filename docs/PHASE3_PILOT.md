# Approved Phase 3 feasibility pilot

This is a separate evaluation of programmed neglect-dependent cue attenuation, not
neural damage, biological trauma, recovery or inherited learning. Core simulation,
training, original checkpoints and Phases 1/2 remain unchanged.

## Fixed protocol

Four counterfactual branches: normal/disrupted childhood × developmental assignment
OFF/ON. Eight stationary resource-neutral background bodies occur in every childhood
branch. A scripted age-120, energy-60 caregiver stays two squares from the newborn.
Care disruption is ON only at pup ages 10–27 inclusive in the disrupted condition.
Crowding energy cost, intrusion and reproduction are disabled in this assay.

Pups receive 59 dependent-care ticks and mature through the original Agent.act at age
60. Normal childhood has no unattended ticks; disrupted childhood has exactly 18.
Peak outstanding neglect is 18, and subsequent attendance clears it by maturation.
The existing formula uses 18/60, assigning impairment 0.30 only to disrupted/ON.
The first adult action uses the frozen RL policy, not the placeholder Brain.

The same subject transfers to an uncrowded world and actually simulates ages 61–120.
Stored impairment is preserved. There are no dependent pups in this bridge, so the
trait cannot affect its inputs. At age 120, position and energy are standardized for
a fresh assigned-pup assessment: energy 20, initial distance four, matched food/nest
geometry, ordinary metabolism and feeding. Care disruption is OFF, background count
zero, reproduction disabled, and every assessment follows parents for all 60 ticks.
Developmental assignment is OFF during assessment; this does not erase the stored trait.

Seed triplets by index j=0,1,2: adult world 400000+j, childhood 410000+j, bridge
420000+j. Each stage uses a separate local Python RNG and seeded CPU Torch fork.
Four trained checkpoints (survival/offspring × seeds 0/1) yield exactly 48 trajectories.
The CLI intentionally has no full-matrix switch. No parameters were altered after results.

## Reproduce the pilot into a NEW directory

```bash
.venv/bin/python -m pytest -q tests/test_developmental_care.py
.venv/bin/python -m pytest -q
.venv/bin/python experiments/developmental_care.py --out results/developmental_care_pilot_reproduction_new
```

Use the CPU environment pinned for Phases 1/2. Existing result directories and nested
outputs inside prior experiment directories are refused. The runner verifies original
checkpoint hashes, finite weights, no gradients/optimizer, and post-evaluation hashes.
Do not use this command to overwrite the completed pilot.

## Measurement and selection controls

Each raw trajectory saves initial state/geometry, all childhood ticks, maturation state,
all bridge ticks and all assessment ticks. Survival to ages 60/120 and adults evaluated
are separate from assigned-pup success. Missing adult assessment is not imputed as an
adult caregiving failure; a separate cohort-recruitment denominator is provided.
Paired adult contrasts are withheld if a branch lacks an assessment.

The three zero-impairment adult branches must match exactly, including actions,
food histories and outcomes. Bridge behavior must match across all four branches.
Assessment initial physics must also match across policy objectives/training seeds.
An unattenuated shallow COPY at each live parent state independently checks that only
pup_dx, pup_dy and pup_need change by clarity. No live subject's impairment is modified
for this audit. Later differences in position, energy, food and need can be consequences
of different actions; they are not held artificially identical across impaired trajectories.

Background bodies never act, eat, age, reproduce, occupy nests or block movement.
Original attendance timing is preserved. The scripted stationary caregiver removes the
movement/snapshot discrepancy during childhood. Transfer is an administrative removal
from the old world, not a death; adult pup assignment is not a reproduction event.

## Pilot results and interpretation

All 48 assigned subjects survived childhood and the bridge; all 48 assessment parents
survived. All 13 technical gates passed. There were no starvation or old-age deaths,
and no offspring independence after parental death. All 12 matched control contexts
and all 2,880 parent sensory audits passed.

Each policy/branch cell has six evaluations. All zero-impairment controls produce
2/6 success for survival policies and 5/6 for offspring policies. Disrupted/ON produces
2/6 and 4/6 respectively. Exactly one matched outcome changes: offspring policy seed 0,
adult world 400001, successful to unsuccessful independence.

Pilot runtime: 4.043 seconds. Tests: 18 new regressions; 265 total passing tests.
The raw dataset has 48 trajectory JSON files, manifest and summary. The separate
analysis directory stores paired differences, integrity checks and the complete report.

Only two independent training pairs are present: no inferential p-values or confidence
intervals are justified as confirmatory evidence. Attendance uses observed care ticks,
which end at pup death or maturation; raw attended ticks are also reported. Behavioral
effects are mixed, including improved attendance in some impaired cases. Do not call
this pilot proof of lasting developmental harm or a reliable policy interaction.

The full planned 1,600-trajectory study would take roughly 135 seconds by direct
extrapolation; allow 2–5 minutes plus validation. It could be informative as a small
policy-sensitivity ablation, including a null result. Persistence remains guaranteed
by the supplied trait rule; only its downstream effect on policy performance is tested.
No larger experiment, new dose, recovery rule or network change has been authorized/run.
