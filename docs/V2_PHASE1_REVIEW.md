# V2 Phase 1 technical review

This is a code audit and implementation handoff, not the university paper or a
literature review. No research training campaign, PPO, or new crowding experiment
has been implemented or run. Smoke outcomes support engineering checks only.

## Preservation and scope

Branch: `dlai-revision-v2`, created directly from completed
`breakdown-mechanisms` HEAD `ce129bef00de069d171c99b797f142f0711fb626`.
The remote branch HEAD was checked read-only before branching. The incoming local
`work` checkout was a merge with that commit as its second parent; it was not used
as the development base. No branch was reset, pushed, or merged. `main` was not
modified. Existing tracked files, including source, runners, tests, results,
checkpoints, archives, and lock/preservation manifests, remain byte-identical.

Only `src/v2/*.py`, `experiments/revision_v2.py`, `tests/test_v2.py`, and this technical
document are added. Evidence and all new runtime outputs are outside `results/`,
under `/workspace/dlai-phase1/`. Outputs use exclusive creation, so a rerun requires
a new directory. Baseline remains reproducible using the original README and
`requirements-reproduction.txt`; original code is not routed through v2.

## Prioritized audit findings

| Priority | Classification and scope | Finding and disposition |
|---|---|---|
| High | Correctness relative to time-discounted credit; shared episodic trainer | If a pup matures after its parent dies, `run_episode` adds the bonus to the dead parent's final reward entry. The intervening world ticks disappear. A deterministic regression demonstrates a tick-3 bonus treated as immediate by legacy credit; v2 discounts it by gamma cubed from tick 0. Original data are not changed. |
| High | Correctness/estimator definition; continuous shared trainer | `run_enclosure.flush` clears trajectories each update window, zero-bootstraps live actors and cannot credit parental actions in earlier windows. Posthumous credit can disappear entirely after a flush. It is a truncated-return approximation, not full-episode REINFORCE. V2 uses full finite tasks and refuses externally truncated MC data. |
| High | Correctness; legacy individual-policy runner, **not final shared checkpoints** | `campaign_individual.run` removes a dead actor's unflushed trajectory and clears its slot. Fatal behavior is omitted and delayed rewards lose the parent identity. Slot reuse also makes a slot ID insufficient for lifetime credit. V2 uses permanent episode IDs and reward-event ownership. This runner is preserved, not repaired in place. |
| High | Objective ambiguity/methodological limitation; shared policies | An action affects other agents, resource availability, births and future actors. Weight sharing does not make own-agent-only returns an unbiased estimator of the colony-total objective. V2 explicitly offers `colony` and `individual_surrogate`; the latter deliberately omits cross-agent score terms. Colony total is a versioned objective choice, not evidence that all historical conclusions are false. |
| Medium | Estimator convention | Legacy uses discounted return-to-go but no outer gamma-to-action-time factor. Combined with a policy-dependent decision-count mean, it is not the literal gradient of expected discounted colony-total return. V2 declares that objective and sums score terms per sampled world, with gamma-to-world-time weighting. Per-decision legacy training remains available unchanged. |
| Medium | Methodological limitation | Batch centering/scaling uses correlated, action-dependent returns and reweights the gradient. Constant-in-batch returns become zero (except the singleton special case). This is not a learned action-independent value baseline. V2 default uses raw MC returns; optional learned baseline uses a separate network trained **after** actor advantages are computed. |
| Medium | Evaluation limitation, already disclosed by original docs | Directional probes condition on moving, ignoring stay/reproduce; a mostly stationary policy can look excellent on the selected movement subset. They are diagnostics, not parenting success. V2 primary logged rates use explicit eligible-birth and care-exposure denominators, with `None` when no opportunities occur. |
| Medium | Exposure limitation, already disclosed | Original assigned-pup trials stop at pup death, so parent survival/meals have unequal follow-up. Natural colony returns are per acting agent and mix different lifetimes. V2 records ticks, fatal adult decisions, surviving adult ticks, births, eligibility, deaths and pup-care exposures separately. It does not silently reinterpret old rates. |
| Medium | Training evidence limitation | Finite loss and a saved final policy do not establish convergence, robustness, or superiority to linear models. Existing 20 final policies and their saved evidence can be verified, but new simulation stability across training seeds requires an approved pilot/campaign. No such claim is made from smoke tests. |
| Low | Intentional retained timing | Batched observations precede `Agent.act` age, costs and maturation. `last_senses` inside act can therefore differ, notably on maturation. Acting order is fixed in the RL loop, not shuffled. Current occupancy is a tick-start snapshot. V2 explicitly preserves this timing rather than changing environment assumptions. |

The categorical log-probability sign, reward-to-go recursion, dead-agent inclusion
in **episodic shared** learning, cap refunds, first mature action recording, and
singleton return handling are correct in the completed baseline. Do not confuse
already-fixed historical defects with new audit findings. Seed isolation also
already exists: `seeded_run` forks CPU Torch state and world RNGs are local. Final
evaluation already uses frozen weights and matched worlds. There is no evidence
of accidental evaluation learning in the completed final protocol. New tests
verify v2 isolation, matched initial hashes, modes and unchanged weight hashes.

Offspring reward is paid only when a live pup becomes independent, not at birth.
Zero dependents produce no bonus; multiple pups can each pay once. A pup dying on
its first independent action does not pay. Posthumous success remains allowed in
v2, with its actual timestamp. These are explicit retained modeling choices.
Programmed crowding costs, proximity care, intrusion, impairment, automatic food
consumption, reproduction thresholds and nest constraints are not removed or
recast as learned biological mechanisms. No new environment mechanism is added.

## Implemented contracts

- `models.py`: actor logits, seeded CPU initialization, local-generator sampling,
  and exact original-state import. Actor counts: linear 78; original MLP 234;
  deep MLP 5,382. Uniform random, randomly initialized neural, hand-designed,
  and previously trained original policies are distinct baseline categories.
- `data.py`, `environment.py`: graph-free, clocked on-policy records, original
  Agent/World dynamics, stable IDs, action/log-probability records, fatal decisions,
  separate externally truncated flags and finite-task boundaries, timed reward
  events, per-cause mortality, birth eligibility and attendance exposure.
  `initial_hash` covers world geometry/resources and founder state before actions.
- `learning.py`: REINFORCE and MC REINFORCE with learned value baseline. The critic
  is a separate 12→64→64→1 network (5,057 parameters), identical across actor
  architectures. It sees pre-action local observations; it is a variance-reduction
  baseline, not a claim of fully observed Markov state. Actor advantages detach
  critic outputs, and the actor update precedes critic fitting to this batch.
  No stale/reused rollout training, implicit normalization, reward clipping,
  optimizer reuse across runs, or PPO update epochs. Entropy regularization and
  gradient clipping are explicit, default disabled, and recorded in configuration.
- `evaluation.py`: common frozen natural-colony protocol with matched world and
  action seeds; state/mode checks, exposure-weighted descriptive aggregation,
  missing opportunity rates represented as undefined, sealed final-test API.
- `experiment.py`: run configuration, training and validation decision counters,
  world ticks, undiscounted reward and discounted colony return, entropy, losses,
  gradient norms, intermediate validation records, checkpoint IDs and JSONL logs.
  Checkpoints save actor/critic/Adam states, next episode index, full seeds/reward/
  environment configuration, dependency versions, CPU thread count, source hashes
  and Git base commit. Restoration rejects source/runtime drift and reproduces
  both evaluation and the next training update. Local generators reconstruct from
  the next episode seed; global RNG snapshots are unnecessary for this CPU path.
- `plotting.py`: actual intermediate observations against cumulative training
  decisions. Undefined rates become gaps, not invented zeros. Single-run plots
  do not create confidence intervals or suggest convergence from two smoke points.
- `experiments/revision_v2.py`: deliberately bounded, two-episode CLI. It does not
  expose a full-campaign or final-test switch. Library APIs support later pilots.

Natural-colony validation retains the training configuration and cap. If no
eligible pups occur, independence remains undefined. A fair parenting study will
also need a prespecified assigned-pup panel (the original assigned-pup runner is
preserved); generalizing that panel across architectures is a next protocol task,
not something inferred from a natural-colony no-birth rate. Existing original
models can be loaded into v2 and evaluated under identical **new** conditions,
but their old training budgets/objectives are not aligned with new learners.

## Exact objective and boundaries

For `colony`, R(t) is the sum of configured survival and own-offspring-independence
payments at world tick t. The finite-task objective is
`J = E[sum_t gamma**t R(t)]`. Each acting agent contributes its score times
`gamma**t * (G_colony(t) - baseline(observation))`. Loss sums scores per world and
averages across independent sampled worlds, not across a random number of agents.
Other actors' future actions are scored too, capturing shared-policy causal terms.
The per-agent value baseline is action-independent at sampling time.

For `individual_surrogate`, only the recipient's events enter its score weight.
Time is still correct and rewards can arrive after the last action. This is an
explicit ablation of credit scope, not an unbiased colony-objective alternative.
Legacy `rl.train` is the original algorithm/estimator; v2 `reinforce` corrects and
declares the estimator while retaining the neural update family. These names must
not conceal a scientific objective change in a results table.

The configured `n_ticks` is a **finite-task horizon**, with no continuation value.
Extinction is an earlier terminal outcome. Agent death is recorded separately
because the colony and posthumous objective can continue after that death.
An external `max_ticks` cutoff is truncation; learners reject it. The independent
return reference requires explicit bootstrap at nonterminal/truncated boundaries.
No bootstrap value has been guessed for delayed posthumous rewards.

## Seeds and budgets

Default reserved namespace:

- Training: `[1000000, 2000000)`, each initialization seed allocated 10,000 world
  seeds (`train_start + initialization_seed*10000 + episode_index`). Architecture,
  reward and algorithm cells use matching seeds within a replicate.
- Validation: `3000000`, `3000001`, `3000002` (technical defaults, not a scientifically
  adequate final validation sample).
- Final test: `[4000000, 4000020)`. No final-test worlds have been executed in Phase 1.
  The API refuses them unless final-test access is explicitly enabled after review.

Initialization, action sampling, world state, training replicate and split are
recorded separately. Fixed validation calls do not consume training RNG streams,
change weights, or count as training interactions. Tests compare final actor
weights under validation frequencies zero, one and two for both algorithms.

`train_to_budget` completes episodes until an adult-decision target is reached.
It reports realized count and overshoot; the latter is less than
`n_ticks * max_population`. It never zero-bootstraps an artificial budget cutoff.
For example, a 500-tick/120-agent task can overshoot by <60,000 decisions. This
requires a prespecified tolerance or learning-curve budget alignment before
calling comparisons strictly matched. Equal episode counts alone are insufficient.
An episode guard raises if extinction/no-decision rollouts prevent the target.
Validation workload, runtime, and training decisions are distinct costs.

## Verification and technical evidence

Original suite before changes: **286 passed**. The final Phase 1 test results and
preservation counts are recorded below after validation. New tests include exact
score gradients, detached advantages, boundary bootstraps, real-clock posthumous
credit, zero/multiple pups, maturation/death, cap refunds, seed isolation, shape/
parameter checks, frozen worlds and weights, denominator accounting, checkpoint
replay/optimizer resume, stale/consumed data, budget guards, and safe outputs.

Two bandit smoke tests (one per learner) use on-policy samples of a one-step,
six-action task with reward only for action 3. Both must improve correct-action
probability above 0.95 in 40 tiny updates. This is a deliberately solvable plumbing
test, not evidence for parenting performance or the need for deep architectures.

Twelve integration cells cover 3 architectures × 2 learners × 2 rewards. Each runs
only two 16-tick episodes with four founders and cap 12, with separate fixed
validation and checkpoint reload. Four baseline adapters are evaluated on matched
8-tick worlds: uniform, untrained MLP, heuristic and original frozen trained MLP.
Checkpoint restoration and plot generation are checked in every integration cell.
No smoke model is promoted as a research result.

Re-run from repository root in the prepared CPU environment:

```bash
export XDG_CACHE_HOME=/workspace/alife-onboarding/cache
export MPLCONFIGDIR=/workspace/alife-onboarding/matplotlib
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv/bin/python -m pytest -q
.venv/bin/python experiments/revision_v2.py \
  --out /workspace/dlai-phase1/new-smoke \
  --architecture deep --algorithm reinforce_value --reward offspring
```

Output directories must be new. The CLI writes a manifest, two checkpoints,
JSONL records with checkpoint SHA-256, and a technical learning-curve PNG.
No process must remain running. Do not run original training runners during setup;
they can overwrite original result paths.

## Readiness, next blockers and proposed comparison

The common CPU foundation and both implemented learners are ready for **review and
an approved small pilot**, not an immediate full scientific comparison. Value
baseline correctness and smoke integration are implemented; remaining work is
multi-seed variance/stability checking, reward scale/critic diagnostics, budget
policy approval, validation-panel size, assigned-care outcome protocol and
prespecified equal tuning budgets. No offspring/survival scientific conclusion
should be selected from technical smoke runs.

PPO remains entirely unimplemented. A separate approved step must specify
next-state/centralized value observations, finite-horizon time features, GAE and
termination versus truncation bootstraps, joint-world versus per-agent advantages,
posthumous events across rollout windows, behavior-policy versions, likelihood
ratios, clipping/KL controls, minibatch/epoch counts, and equivalent data budgets.
Current records have old action log probabilities and stable IDs, but are full
MC episodes; they are not a drop-in per-agent PPO rollout buffer.

Minimum proposed primary matrix, before PPO:

| Learner | Linear 12→6 | Original 12→12→6 | Deep 12→64→64→6 | Rewards |
|---|---|---|---|---|
| V2 REINFORCE | train | train | train | survival; survival + offspring independence |
| V2 REINFORCE + learned value | train | train | train | survival; survival + offspring independence |

That is **12 cells**, with preferably 10 independent initialization/world-stream
replicates per cell (120 models), same environment version, explicit colony scope,
paired world panels, declared interaction budget and equal tuning allowances.
Evaluate uniform random, random neural initializations for all three architectures,
and the heuristic as separate untrained controls. Evaluate original frozen MLPs
as historical transfer references. Retrain the original legacy estimator under
aligned budgets only if it is to become a primary trained comparator; it is not
automatically matched to either v2 learner. Add PPO only after its separate review.

Technical throughput measurement: one 64-tick/16-founder UNIVERSE_25 rollout and
one update per architecture/learner, CPU with two Torch threads. Each cell used
1,024 training decisions; measured throughput was approximately **13.5k–16.4k
adult decisions/second**, excluding initialization, validation, checkpoint I/O
and plotting. These short low-density runs underestimate later costs and are not
production benchmarks. At that rate, 1M decisions/model has a raw floor of about
61–74 seconds; 120M decisions across the proposed matrix has a raw floor of
2.0–2.5 hours on this machine. Budget provisionally **6–12 wall-clock hours** for
120 models at 1M decisions each, plus separately measured validation/tuning costs;
this is an uncertain planning allowance, not a convergence recommendation.
For 10M/model, scale the training component about tenfold. A long 500-tick episode
with a growing population, larger validation panels, or inefficient critic fits
can change throughput substantially. Profile an approved pilot before reserving a
campaign. CPU scalar simulation dominates enough that GPU speedup is unproven.

Phase 1 stops here for review. No push, merge, paper, literature review, full
training campaign or new crowding experiment is authorized by this handoff.

## Final executed Phase 1 results

- Complete suite: **346 passed** in 9.50 seconds: original 286 plus 60 new cases.
- Both synthetic bandit learners exceeded the required 0.95 action-success threshold.
- All twelve bounded integration cells passed; all four baseline adapters shared
  identical initial-world fingerprints. Intermediate checkpoints reloaded exactly.
- Six stochastic seeded scenarios matched the original collector's complete
  observation stream, decision counts, births, final population and total payments.
  This isolates the new credit/objective estimator from unintended dynamics changes.
- Original saved-evidence verifier passed without retraining or evaluation replays:
  20 checkpoints, 1,200 original training-world seeds, 800 assigned-pup records,
  400 natural-colony records and 3,200 controlled-care records verified.
- Preservation SHA-256 check: **256/256 original tracked files byte-identical**,
  including **178 original result/artifact files**. HEAD remains the preserved
  baseline commit on `dlai-revision-v2`; changes are new, uncommitted files.
- Final-test panel untouched. Full campaigns, PPO and new crowding experiments
  were not run. Nothing was pushed or merged.

Evidence paths in this prepared workspace:
`/workspace/dlai-phase1/preservation.json`, `legacy-verification.json`,
`throughput.json`, and `integration-final/integration.json`. The integration
subdirectories contain manifests, two checkpoints each, validation logs and plots.
Source/runtime hash enforcement means earlier development checkpoints can correctly
refuse restoration after a source edit; use the final integration outputs above.
