# Full frozen neural-policy evaluation

Completed once on breakdown-mechanisms, base3c6a601, with no retraining or core edits.
Original worlds100000–100019 × energies10/20 × ten paired seeds × five uniqueconditions
=2,000 trials. Ten distinct verified initial networks; twenty frozen trained networks.
Normal trained trials are shared between the training and ablation comparisons.
Evaluation loop runtime35.922seconds; no failed or incomplete final trials.

## Findings for the three-page report

Offspring-reward gradient training increased assigned-pup independence from19.75%
for the exact shared initializations to50.75%: +31pp, unadjusted95% paired interval
+21.5 to+39.25pp, exact two-sided sign-flip p0.00390625, Holm p0.01171875.
Nine seeds improve and one ties; none worsen. Zeroing the three own-pup inputs reduces
this trained-policy success to4.5%: −46.25pp, interval−53 to−39.25pp, exact p0.001953125,
Holm p0.0078125, with negative effects in every seed.

Survival training gives9.75% independence (−10pp versus initialization), with seven
negative, one zero and two positive seed effects. Its unadjusted interval is−17.25
to−2.5pp, but Holm p0.0859375 does not pass the four-test family threshold0.05.
Survival masking gives5.5% (−4.25pp), interval−9 to−0.5pp, exact/Holm p0.09375.
An unadjusted bootstrap interval excluding zero and a nonsignificant discrete
sign-flip/Holm result are compatible; do not select the more favorable test.

Inference uses ten independent training-seed pairs, not2,000 independent networks.
Bootstrap100,000 paired resamples, seed20261014; all1,024 sign patterns; Holm across
four prespecified comparisons. Intervals are unadjusted percentile intervals conditional
on this fixed twenty-world panel. The exact test assumes null sign exchangeability.
Original held-out worlds were reused, including the prespecified pilot subset;
this is not a new untouched test set. No protocol changes followed pilot results.

This provides matched-initialization evidence of objective-specific improvement
from the saved gradient-trained policies and functional dependence on own-pup sensory
information. It does not establish that survival training improves caregiving or
that more network depth is necessary. Grouped zero masking combines direction/need
removal and induces an input-distribution shift; it does not identify semantic hidden
representations. There is no training or weight change during these evaluations.

## Mortality and follow-up

All failed pups die of neglect. All parent deaths are starvation and occur atenergy10.
Posthumous successes: initial24/79, survival5/39, offspring57/203, survival masked5/22,
offspring masked5/18. These remain included in the prespecified independence outcome
and do not establish continuous parental care. Adult-alive success counts, descriptive
only:55,34,146,17,13 respectively. Atenergy20 all parents survive, and rates are21.5%,
10%,53%,6%,4% respectively: the main qualitative pattern persists without parental death.

The original assay stops at pup death. Thus parent survival, meals and attendance
have outcome-dependent exposure durations. Do not interpret raw parent deaths or
meal counts as directly comparable60-tick risks/rates. Full-survival/early-stop/death
statuses and observed care-tick denominators are saved, not hidden. Parental-care timing,
truncated/posthumous reward credit and dead-individual trajectory limitations are
unchanged; the latter is not the training mode evaluated here.

## Artifacts, verification and commands

Raw4files: results/neural_policy_v1/{manifest,records,summary,INTEGRITY}.json (93.30MB).
Analyses: results/neural_policy_v1_analysis/analysis.json, training_seed_results.csv,
REPORT.md, CAPTION.md, neural_policy.pdf (vector), neural_policy.png (300dpi),
verification.json, preservation_receipt.json and pilot_runner_source.txt.
The existing pilot files are unchanged. The source snapshot preserves its exact
pre-extension runner. Archive the93MB raw dataset losslessly at preservation review;
do not add uncompressed duplicates or casually use `git add .`.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python experiments/verify_neural_policy.py --replay --out work/neural_verification_new.json
.venv/bin/python experiments/analyze_neural_policy.py --out work/neural_figures_new
```

Run from repository root in recorded CPU environment, usingNEW output paths (create
work/ beforehand). Current full runner reproduction command, documentation only:
`.venv/bin/python experiments/neural_policy_evaluation.py --full --out results/neural_reproduction_new`.
No additional full run is needed for review. No checkpoint fitting occurs in anycommand.
Dependencies are already pinned in requirements-reproduction.txt. Installation in a
new environment has not been tested. Matplotlib issued a font-cache warning during
initial generation; reviewed PNG/PDF rendered correctly and output generation succeeded.

All286 tests pass preflight and after the run. Independent checks cover all2,000
unique keys,800 exact historical trained replays,71,024 input audits, twenty checkpoint
hashes, ten distinct initial hashes, mortality/population accounting and all four
bootstrap/exact/Holm results. One hundred additional frozen trials replay exactly.
All earlier tracked and untracked result files were hashed before execution and remain
unchanged. main, Git HEAD and index remain unchanged. Nothing committed or pushed.

## Architecture baseline recommendation

Do not add a trained linear baseline to the main three-page report by default.
The new trained-versus-initial and sensory-ablation figure directly strengthens the
learning evidence; Phases1/2 remain main scientific results, with Phase3 compact support.
The existing model is a one-hidden-layer12→12 tanh→6 MLP (234parameters), not a large
or deeply stacked network. A12→6 linear policy has78parameters. Only training it with
matched objectives, seeds, optimizer/budget and evaluations would fairly test whether
nonlinear capacity adds value. Untrained linear versus trained MLP is not a valid
architecture comparison. Such training is not authorized now and creates another
experiment/inference family and validation burden. Pursue it only if architecture
necessity is a stated grading requirement and enough time remains to report null or
superior-linear results honestly; otherwise preserve and write the report now.
