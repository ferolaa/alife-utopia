# Neural-policy pilot: descriptive feasibility results

Two training seeds, three prespecified original held-out worlds, both parent energies; 60 unique trials. No full matrix run.

| Cell | Energy | Independent / trials | Attendance (%) | Meals | Parent deaths | Alive early stop | Survived full trial | Posthumous independence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | pooled | 2/12 | 22.77 | 6 | 1 | 9 | 2 | 0 |
| survival_normal | pooled | 0/12 | 19.43 | 21 | 0 | 12 | 0 | 0 |
| offspring_normal | pooled | 12/12 | 73.31 | 6 | 4 | 0 | 8 | 4 |
| survival_masked | pooled | 0/12 | 9.19 | 24 | 0 | 12 | 0 | 0 |
| offspring_masked | pooled | 0/12 | 17.07 | 30 | 0 | 12 | 0 | 0 |
| initial | 10 | 1/6 | 22.77 | 3 | 1 | 4 | 1 | 0 |
| survival_normal | 10 | 0/6 | 19.57 | 10 | 0 | 6 | 0 | 0 |
| offspring_normal | 10 | 6/6 | 62.99 | 2 | 4 | 0 | 2 | 4 |
| survival_masked | 10 | 0/6 | 9.63 | 12 | 0 | 6 | 0 | 0 |
| offspring_masked | 10 | 0/6 | 17.70 | 14 | 0 | 6 | 0 | 0 |
| initial | 20 | 1/6 | 22.77 | 3 | 0 | 5 | 1 | 0 |
| survival_normal | 20 | 0/6 | 19.30 | 11 | 0 | 6 | 0 | 0 |
| offspring_normal | 20 | 6/6 | 83.62 | 4 | 0 | 0 | 6 | 0 |
| survival_masked | 20 | 0/6 | 8.74 | 12 | 0 | 6 | 0 | 0 |
| offspring_masked | 20 | 0/6 | 16.42 | 16 | 0 | 6 | 0 | 0 |

Paired independence differences by training seeds0,1 (percentage points):
- survival_training: -33.33, +0.00
- offspring_training: +66.67, +100.00
- survival_masking: +0.00, +0.00
- offspring_masking: -100.00, -100.00

All parent deaths were starvation, all at energy10. Four offspring-policy independence successes occur after parent death. Early stopping makes raw survival/feeding totals exposure-dependent. Attendance is pooled over observed care ticks, not an independent-tick estimate.

Masked-policy same-state diagnostics: survival mean KL0.0694 nats, argmax changes43.24%; offspring KL0.1318, argmax changes69.27%. These summarize policy calls, not independent replicates. Normal inputs/logits have exact zero change.

Interpretation: this small panel demonstrates technical feasibility and potentially informative learning/input sensitivity. It does not establish general improvement over initialization, general sensor dependence, neural representations or full-population effects. Survival-trained failures and survival-masking floor remain visible.

The earlier no-op assertion initially compared JSON lists against in-memory tuples; comparisons now use JSON-normalized outcomes. No dynamics, parameters, seed choice or saved prior results changed. All60 final trials replay exactly. See docs/NEURAL_POLICY_PROTOCOL.md for full prespecified comparisons and limitations.
