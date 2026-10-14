# Full Phase 3: programmed neglect-dependent cue attenuation

1,600 complete trajectories; runtime 117.49 seconds. Preflight: 265 tests passed.

| Objective | Childhood | Assignment | Successful / assessed | Attendance | Meals | Neglect deaths |
|---|---|---|---:|---:|---:|---:|
| survival | normal | OFF | 22/200 (11.00%) | 22.29% | 891 | 178 |
| survival | normal | ON | 22/200 (11.00%) | 22.29% | 891 | 178 |
| survival | disrupted | OFF | 22/200 (11.00%) | 22.29% | 891 | 178 |
| survival | disrupted | ON | 20/200 (10.00%) | 21.70% | 979 | 180 |
| offspring | normal | OFF | 114/200 (57.00%) | 54.59% | 352 | 86 |
| offspring | normal | ON | 114/200 (57.00%) | 54.59% | 352 | 86 |
| offspring | disrupted | OFF | 114/200 (57.00%) | 54.59% | 352 | 86 |
| offspring | disrupted | ON | 84/200 (42.00%) | 44.62% | 515 | 116 |

| Planned contrast | Effect (pp) | 95% paired interval | Exact p | Holm p |
|---|---:|---|---:|---:|
| survival | -1.00 | -4.50, 1.50 | 1.000000 | 1.000000 |
| offspring | -15.00 | -21.00, -9.50 | 0.001953 | 0.005859 |
| objective_interaction | -14.00 | -20.50, -8.50 | 0.001953 | 0.005859 |

## Individual training-seed results

| Seed | Survival control | Survival impaired | Survival effect pp | Offspring control | Offspring impaired | Offspring effect pp | Interaction pp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.0 | 10.0 | 0.0 | 90.0 | 65.0 | -25.0 | -25.0 |
| 1 | 5.0 | 10.0 | 5.0 | 55.0 | 50.0 | -5.0 | -10.0 |
| 2 | 10.0 | 10.0 | 0.0 | 60.0 | 25.0 | -35.0 | -35.0 |
| 3 | 25.0 | 10.0 | -15.0 | 65.0 | 45.0 | -20.0 | -5.0 |
| 4 | 10.0 | 10.0 | 0.0 | 55.0 | 50.0 | -5.0 | -5.0 |
| 5 | 15.0 | 15.0 | 0.0 | 65.0 | 60.0 | -5.0 | -5.0 |
| 6 | 10.0 | 10.0 | 0.0 | 50.0 | 30.0 | -20.0 | -20.0 |
| 7 | 15.0 | 15.0 | 0.0 | 30.0 | 15.0 | -15.0 | -15.0 |
| 8 | 10.0 | 10.0 | 0.0 | 55.0 | 45.0 | -10.0 | -10.0 |
| 9 | 0.0 | 0.0 | 0.0 | 45.0 | 35.0 | -10.0 | -10.0 |

Frozen policies; childhood exposure fixed at 18 unattended ticks. Adult assessment: age 120, energy 20, no background bodies or active care disruption. Each cell has 200 evaluations but inference uses ten training-seed pairs. Three zero-impairment controls are identical. Diamonds show paired means, small points individual seed effects; error bars are unadjusted 95% paired bootstrap intervals conditional on the fixed world panel. Negative contrasts indicate fewer independent assigned pups. Persistence and attenuation are programmed, not learned neural damage.

100000 paired seed bootstrap resamples; RNG 20261014; unadjusted percentile 95% intervals conditional on fixed twenty-world panel; all 1024 exact two-sided sign flips; Holm across only three planned contrasts

Interpretation must distinguish programmed persistence from its measured downstream effect on policy performance. Missing adult assessments cannot be silently classified as adult care failures. Observed-care attendance ends at pup death/maturation; mortality uses full parental follow-up. Modelled cue scaling is not trauma, biological recovery, inherited learning or changed network weights. The first three worlds and first two training pairs were used in the pilot; no parameters changed afterward.
