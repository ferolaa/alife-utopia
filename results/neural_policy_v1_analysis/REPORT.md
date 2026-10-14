# Frozen neural-policy evaluation

| Policy | Energy | Independence | Attendance | Meals | Neglect deaths | Parent deaths | Alive early stop | Full survival | Posthumous success |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| initial | pooled | 79/400 (19.75%) | 27.69% | 293 | 321 | 56 | 289 | 55 | 24 |
| survival_normal | pooled | 39/400 (9.75%) | 18.44% | 887 | 361 | 9 | 357 | 34 | 5 |
| offspring_normal | pooled | 203/400 (50.75%) | 44.50% | 405 | 197 | 80 | 174 | 146 | 57 |
| survival_masked | pooled | 22/400 (5.50%) | 14.58% | 1031 | 378 | 9 | 374 | 17 | 5 |
| offspring_masked | pooled | 18/400 (4.50%) | 12.59% | 863 | 382 | 7 | 380 | 13 | 5 |
| initial | 10 | 36/200 (18.00%) | 25.70% | 143 | 164 | 56 | 132 | 12 | 24 |
| survival_normal | 10 | 19/200 (9.50%) | 17.88% | 445 | 181 | 9 | 177 | 14 | 5 |
| offspring_normal | 10 | 97/200 (48.50%) | 40.18% | 190 | 103 | 80 | 80 | 40 | 57 |
| survival_masked | 10 | 10/200 (5.00%) | 13.93% | 513 | 190 | 9 | 186 | 5 | 5 |
| offspring_masked | 10 | 10/200 (5.00%) | 12.66% | 430 | 190 | 7 | 188 | 5 | 5 |
| initial | 20 | 43/200 (21.50%) | 29.68% | 150 | 157 | 0 | 157 | 43 | 0 |
| survival_normal | 20 | 20/200 (10.00%) | 18.99% | 442 | 180 | 0 | 180 | 20 | 0 |
| offspring_normal | 20 | 106/200 (53.00%) | 48.81% | 215 | 94 | 0 | 94 | 106 | 0 |
| survival_masked | 20 | 12/200 (6.00%) | 15.23% | 518 | 188 | 0 | 188 | 12 | 0 |
| offspring_masked | 20 | 8/200 (4.00%) | 12.53% | 433 | 192 | 0 | 192 | 8 | 0 |

| Contrast | Difference pp | 95% interval pp | Exact p | Holm p | Negative / zero / positive seeds |
|---|---:|---|---:|---:|---|
| survival_training | -10.00 | -17.25, -2.50 | 0.042969 | 0.085938 | 7 / 1 / 2 |
| offspring_training | +31.00 | +21.50, +39.25 | 0.003906 | 0.011719 | 0 / 1 / 9 |
| survival_masking | -4.25 | -9.00, -0.50 | 0.093750 | 0.093750 | 6 / 3 / 1 |
| offspring_masking | -46.25 | -53.00, -39.25 | 0.001953 | 0.007812 | 10 / 0 / 0 |

Two thousand unique trials; ten unique initial networks and twenty trained networks. Shared normal trained and initial controls are not counted as independent copies.

ten paired training seeds; 100000 paired percentile bootstrap resamples, RNG 20261014; 1024 exact two-sided sign flips; Holm across four prespecified contrasts; fixed-world conditional unadjusted 95% intervals

Care attendance is pooled over observed care ticks; meals and parent survival have unequal follow-up because the original protocol stops at pup death. Parent deaths and posthumous successes remain explicit. This measures gradient-training-associated caregiving performance and functional sensitivity to removed own-pup information, not hidden representations. Masking induces an input-distribution shift. No training occurred during evaluation.
