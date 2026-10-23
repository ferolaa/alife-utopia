# Controlled v2 learning pilot: frozen protocol

This is a technical protocol and implementation record, not the university paper.
It authorizes only the eight original-MLP learning runs requested by the user.
No scientific setting will be tuned in response to outcomes. A null result is valid.
Corrections and pilot outputs remain uncommitted until user review.

## Primary configuration

- Branch/base: `dlai-revision-v2`, `66fe988f5774c744f245db2270278fb69747896a`.
- Environment: exact preserved UNIVERSE_25 training preset, 500 ticks, 40 founders,
  cap 120. This is the historical episodic training preset, not the long
  12,000-tick continuous-colony configuration. Food, sensing, care, maturation,
  ageing and the existing 0.025 crowding energy cost remain unchanged. Care
  blocking, nest intrusion and developmental impairment remain disabled.
- Actor: 12→12→6 tanh MLP, 234 parameters. No linear/deep study runs.
- Algorithms: v2 REINFORCE; v2 MC REINFORCE with a separate learned value baseline.
  Critic: 12→64→64→1 tanh, 5,057 parameters. Separate Adam optimizer and MSE log.
- Rewards: colony-total, gamma 0.99. Both objectives pay 0.01 per surviving adult
  action; the offspring condition additionally pays 1.0 per live own-pup
  independence event, preserving actual event times and posthumous credit.
- Adam learning rates: actor 0.001 and critic 0.001, unchanged v2 defaults.
  No return normalization, entropy bonus or gradient clipping. One actor update
  per completed episode; one separate critic update when enabled.
- Initialization/training seeds: 0 and 1, paired across algorithms/objectives.
  Training worlds: 1,000,000 + initialization_seed×10,000 + episode index.
- Exactly 2 algorithms × 2 objectives × 2 seeds = 8 runs.
- Target: 200,000 sampled adult training decisions/run, fatal actions included.
  Complete the final episode; never silently truncate MC experience.
- Prespecified tolerance: at most 20,000 excess decisions/run and at most a
  20,000-decision realized-budget spread across runs. Exceeding either limit stops
  the pilot for approval. Realized budgets and optimizer updates are reported.
  The theoretical one-episode bound is <60,000, so the tolerance is checked against
  actual completed episodes, not assumed to hold.
- Initial validation, then thresholds 50k, 100k, 150k and 200k. Evaluate at the first
  completed episode boundary reaching a threshold. Record actual decision counts;
  one boundary panel can represent multiple crossed thresholds. No repeated
  optimization steps are introduced to reach an artificial evaluation point.

## Fixed evaluation panels

Five new validation-world seeds: 3,000,100 through 3,000,104. They are outside the
training block and reserved final-test block. No final-test worlds are instantiated
or inspected. The validation panel is descriptive, not a sufficiently large final
research panel. The exact same worlds and action seeds recur for all policies.

Natural-colony panel: five unchanged 500-tick training-preset worlds, no training.
Record births, late-birth exclusion, eligible-offspring independence, attendance,
exposures, mortality, action counts and population histories. Zero opportunities
produce undefined rates, not artificial zero performance.

Assigned-pup panel: five worlds × parent energies 10 and 20 = ten trials/panel.
Each trial supplies one mature parent (age 120) and one own newborn in a claimed
nest. Choose a nest whose closest initial food is at Chebyshev distance 4..8, and
place the parent four cardinal squares away where food is visible. Candidate order
is deterministic, and the world RNG selects the candidate before policy sampling.
Energy conditions retain the same geometry and action RNG seed. No further births
are possible (`reproduce_threshold=inf`); a reproduce action is a wasted action.

All trials follow outcomes for the full dependency horizon of 60 ticks. Pup death
does not end parent follow-up. If both subjects die, the remainder is explicitly
absorbing and analytically ascertained, rather than simulated as living exposure.
The primary assigned-pup outcome is independence: alive after its first independent
adult action. Posthumous independence is counted separately. Attendance is
attended pup-care ticks / actual pup-care ticks, including fatal dependent ticks;
there are at most 59 care ticks before maturation. Parent at-risk ticks, death
times, simulated ticks, absorbing ticks and complete outcome follow-up are separate.
The assigned-trial denominator is NEVER pooled with natural eligible births.

Policies are frozen; evaluation mode is restored. Local action generators isolate
validation from training RNGs. Initial conditions and policy hashes are recorded.
Every model is compared with its exact actor initialization. The colony-survival
condition may indirectly favor raising offspring, so it is not a condition with
no possible parental incentive. An assigned-pup score of zero is not itself proof
of no natural-colony care; both outcomes are retained.

## Corrections and evidence

`src/v2/environment.py` gains observational callbacks and action/failure counts;
Agent/World mechanics remain untouched. `src/v2/care.py` implements the assay.
`Experiment` gains a cumulative-decision scheduler, initial validation, separate
assigned panels and persisted scheduler state. The old episode-frequency API is
retained only for Phase 1 compatibility; this pilot explicitly disables it.
Checkpoint schema 2 and the collector/assay version identify these corrections.
Old source-mismatched checkpoints are not silently migrated.

`src/v2/pilot.py` and `experiments/pilot_v2.py` freeze the eight-run matrix, refuse
existing output paths, snapshot code/protocol, enforce budget tolerances, write
complete episode/validation logs and checkpoint hashes, replay final checkpoints,
and generate per-run and per-seed overview plots. No hyperparameter search exists.

New tests include an independently enumerated two-agent expected-return gradient.
One agent's action changes the other's later reward; Monte Carlo joint scores
approach the exact gradient. Own-only credit and missing outer time-discount
weights both fail the test. Further tests cover threshold grouping, full parent
follow-up, absorbing death, posthumous success, assay reproduction suppression,
matching environments, frozen policies, scheduled-validation checkpoint recovery,
budget overshoot, and actor/critic RNG isolation.

Before launch, run the complete existing and new suite, compare every historical
file with the preserved baseline Git object, and verify no previous output path
will be overwritten. Launch only after those checks pass. Source/protocol contents
are then frozen for the eight runs. No source changes during collection.

Output root: `/workspace/dlai-pilot-v2-001` (must not exist).

```bash
export XDG_CACHE_HOME=/workspace/alife-onboarding/cache
export MPLCONFIGDIR=/workspace/alife-onboarding/matplotlib
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv/bin/python -m pytest -q
.venv/bin/python experiments/pilot_v2.py --out /workspace/dlai-pilot-v2-001
```

No p-values or statistical-significance claims are justified by two training seeds.
Review learning trajectories, exact paired changes, per-energy outcomes, failures,
critic loss and gradient/entropy diagnostics before proposing another experiment.
No push, commit, PPO, new crowding experiment or full research campaign occurs.
