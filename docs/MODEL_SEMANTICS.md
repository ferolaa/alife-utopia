# Model semantics and scientific boundaries

The completed project uses frozen shared PyTorch policies for evaluation. Older
individual-policy, evolutionary and exploratory scripts remain for provenance;
they are not substitutes for the final matched experimental protocols.

## Environment and tick order

UNIVERSE_25 is a walled 48×48 grid. Four nonoverlapping, fully contained 7×7 feeder
regions give exactly 196 legal food squares, with 60 immediately replenished food
items (30.6% occupancy). Seeded constructive placement uses a randomly offset tiling;
it is not uniform over every disjoint geometric layout. Nests are scattered, not
necessarily on the perimeter. Food/nest overlap and access distances still vary.
Infeasible food requests raise an error rather than silently placing fewer items or
expanding beyond configured feeders; replenishment respects feeder_spread.

An adult tick advances age, settles maturation and finished nest occupancy, spends
metabolic/crowding energy, acts, eats on its resulting square, then checks starvation
and old age. Eating can rescue an energy deficit, but cannot avert old-age death.
Agent.act is the authoritative implementation. Pup dependency ends at age 60;
reproductive maturity starts at age 120. The first independent action uses the RL
policy. Neighbour inputs use the current agent position against the tick-start
occupancy snapshot. Attendance uses parent positions after movement; crowd-based
care blocking/intrusion use tick-start crowding. This retained timing is a limitation,
not a silently corrected assumption.

## Training and evaluation

PolicyNet has 12 inputs, 12 tanh hidden units and 6 action logits (234 parameters). The
categorical stochastic policy is optimized by REINFORCE with discounted returns,
autograd backpropagation and Adam. Survival reward is 0.01 per surviving adult tick;
offspring reward adds 1.0 at own-pup independence. Final training uses ten paired seeds,
120 episodes per model,500 ticks per episode, gamma 0.99 and lr 0.01. Initial weights and
training worlds are matched within objective pairs; episode seed=training_seed×1000+
iteration. Evaluation worlds and action RNG seeds are matched; weights are frozen.

Phase 1 combines natural-colony observations with assigned-pup care trials.
Phase 2 isolates care blocking: stationary background bodies contribute to crowding
but do not feed, breed, occupy nests or consume resources. Crowding energy penalties
are disabled in this assay, ordinary metabolic costs remain. Phase 3 scripts matched
childhood attendance, assigns persistent impairment 0 or 0.30, matures subjects to120,
and transfers them to identical uncrowded parenting trials. Only three own-pup input
channels are attenuated; persistence is a programmed trait, not altered network weights.
The neural analysis uses exact shared initial networks and masks those three channels
at evaluation only. Normal trained evaluations exactly replay Phase 1; these reused
trials should not be portrayed as independent corroboration.

## Interpretation and provenance

Independent assigned-pup success is the prespecified outcome, not uninterrupted
attendance. Posthumous successes are reported. Original Phase 1/neural trials stop at
pup death; parent mortality/meal totals therefore have unequal follow-up. Phase 2/3
follow parents for the full 60 ticks. Natural-colony food access, starvation, ageing,
breeding opportunities and population caps prevent attributing all demographic
changes to crowding alone. No experiment tunes parameters to force collapse.

Parental care is a proximity rule. Programmed disruption, sensory attenuation and
nest intrusion are model mechanisms inspired by hypotheses, not measured biology.
The final controlled assays do not establish a biological reproduction of Calhoun,
trauma, inherited learning, neural damage, genuine recovery or necessary depth.
Input masking induces a distribution shift and does not decode hidden representations.

Ten training-seed pairs are the independent inferential units; repeated worlds,
energies, agents and ticks are not extra network replicates. Bootstrap intervals
condition on the fixed world panels and are unadjusted. Exact sign-flip tests assume
null sign exchangeability. Holm corrections apply within specified families, not
across every exploratory result in the repository. Survival training's adjusted
initialization comparison is nonsignificant. Delayed/posthumous reward credit and
dead-individual trajectories in individual-policy training remain documented limitations;
the final checkpoints were trained with episodic shared policies, not individual learning.

See phase-specific reviews and manifests for complete protocols, denominators,
seed lists and statistics. All completed data/checkpoints are preserved separately
from analysis outputs. The original Phase 3 expected-count metadata defect is unchanged;
its explicit correction and executed-source snapshots establish the actual 1,600-trajectory matrix.
