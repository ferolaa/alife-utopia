# Frozen neural-policy comparison: pilot and proposed full evaluation

Base commit: 3c6a601. Evaluation only; no new training, architecture changes, core edits,
or revision of Phases 1–3. Runner: experiments/neural_policy_evaluation.py.

## Initialization and intervention

Reconstruct PolicyNet(seed=s): fork the CPU Torch generator, manual_seed(s), create
12→12 tanh→6 linear-score network in its original constructor order. src/rl.py train
uses exactly this constructor. Compare parameter-byte SHA-256 with BOTH saved
initial_policy_hash fields for all seeds 0–9 before any trial. All ten match and are
distinct. Frozen trained checkpoint parameters also match original trained hashes.
Ten unique initial networks are shared across the two objectives, not twenty initial
replicates. The network has one hidden layer; do not describe it as a large/deep model.

Delegate all dynamics to final_rewards.assigned_pup_trial unchanged: original held-out
seeds 100000–100019, energies 10/20, supplied dependent pup, parent age120 at distance4,
60 planned ticks, unlimited food, original metabolic/crowding costs and care timing,
no reproduction. Policies control the parent and any pup taking its first adult action,
exactly as in the original protocol. Parent/pup brains, lifespan draws and geometries
use the evaluator's seeded local RNG. CPU action sampling is seeded per world and
isolated from unrelated global RNG. There is no optimizer or weight update.

ObservedPolicy clones each input tensor and zeros ONLY channels9–11 (pup_dx,pup_dy,
pup_need). Caller tensor and first nine inputs stay unchanged. Reference probabilities
at the SAME physical state require no action sampling or extra RNG draws. Normal
wrapper outputs are a no-op; all pilot outcomes match historical saved outcomes after
ordinary JSON tuple/list normalization. Diverged masked trajectories need not visit
baseline states: the same-state comparison is within each captured inference call.
Policies themselves remain frozen; zeroing inputs does not expose representation
semantics or change weights. Input masking induces a distribution shift.

## Prespecified pilot

Training seeds0,1; original evaluation worlds100000–100002; both energies.
Five unique cells: initial, survival normal, offspring normal, survival masked,
offspring masked =60 trials. Normal trained trials are reused between experiments,
not independently resampled or counted twice. No seeds/parameters were chosen after
seeing outcomes. All ten initialization checks precede this subset evaluation.

```bash
.venv/bin/python -m pytest -q tests/test_neural_policy_evaluation.py
.venv/bin/python experiments/neural_policy_evaluation.py --out results/neural_policy_pilot_new
```

Every output must be NEW. The full matrix was subsequently approved and executed with --full; no scientific parameters changed. Saved manifest
records versions, exact seeds, checkpoint/initial hashes and runner/evaluator hashes;
INTEGRITY.json hashes the unchanged records, manifest and descriptive summary.

## Prespecified full matrix and inference — subsequently completed

All ten paired seeds × twenty ORIGINAL held-out worlds × energies10/20 × fivecells
=2,000 unique trials. Each cell has400 trials and each training seed40 trials percell.
Experiment A separately contains1,200 trials; B1,600, sharing800 normal trained trials.
Thirty unique networks: ten initial, ten survival-trained, ten offspring-trained.
Worlds/energies are repeated evaluations, not independent training replicates. This
extends evaluation on the established held-out panel; it is not a new untouched test
set following pilot exploration. Report that reuse transparently.

Primary outcome: supplied pup survives to independence. Four planned contrasts:
1. survival normal minus shared initial;
2. offspring normal minus shared initial;
3. survival masked minus survival normal;
4. offspring masked minus offspring normal.

For each contrast calculate40-trial success fraction per training seed, then paired
seed differences. Use ten seed pairs,100,000 paired bootstrap resamples(RNG20261014),
unadjusted percentile95% intervals conditional on fixed twenty-world panel, all1,024
exact two-sided sign flips, and Holm correction across these four planned tests.
The sign-flip test assumes null sign-exchangeability; sharing initial controls makes
contrasts correlated but does not invalidate Holm's multiplicity protection. Equal
energy weighting is fixed in advance. Energy-specific effects/rates are descriptive
secondary analyses; do not select one energy as primary after inspection.

Secondary: care attendance (attended/observed care ticks and raw tick totals), parent
meals, parent death causes, survival/early-stop status, posthumous independence. For
attendance inference, average per-trial fractions within each training seed rather
than treating ticks as replicates; pooled tick fractions are descriptive only. Same-
state probability KL and argmax change are mechanism diagnostics over recorded policy
calls (including a pup's first adult action), not independent statistical samples or
claims about hidden representations. Clinical-sounding trauma/resilience claims are
not appropriate. Masking may remove directional AND urgency information simultaneously.

Early stopping at pup death censors parental follow-up: raw parent-survival percentages
and meals are not comparable full60-tick risks/exposure counts. Report full-survival,
alive-at-early-stop and death separately. Do not change this approved original protocol
to conceal the issue. Independence after parent death is recorded, not removed.

## Feasibility decision

Pass criteria, regardless of effect sign: exact initialization/all frozen hashes;
normal historical replay; identical starting worlds; mask-only channel changes;
RNG isolation; population/death accounting; exact replay; complete matrix; no prior
artifact changes. Pilot passes. Null or worse-than-untrained learning remains useful:
survival training does not target caregiving. Survival masking has a floor here, so
its zero outcome difference is not evidence of input independence. Proceed if approved.

Pilot evaluation loop took1.179s for60 trials. Linear extrapolation gives39.3s for2,000;
budget1–3minutes for checkpoint verification, serialized audits and analysis on this
CPU, plus tests/replay verification. This is an estimate, not a full-run measurement.
No retraining or architectural baseline is needed for these comparisons.
