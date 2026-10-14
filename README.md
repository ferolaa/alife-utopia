# alife-utopia

An artificial-life simulation for the DLAI (Deep Learning & Applied AI) course project,
Sapienza University of Rome.

## Question

Calhoun's "Universe 25" experiment found that a rodent population given unlimited food
still collapsed and died out as density rose. This project recreates that enclosure in
simulation: a walled pen with food at fixed feeders, nest boxes around the perimeter,
creatures that age, and young that die unless a parent stays close. The creatures are
driven by small neural networks trained by reinforcement learning in the same pen while
the population is held low, so they arrive competent at feeding themselves and raising
young, as Calhoun's mice did. The cap is then lifted and the pen fills up while the
networks go on learning, with the reward and the environment held fixed so that density is
the only thing that changes. Mechanisms Calhoun described are switched on one at a time to
see which of them a competent population can absorb and which it cannot.

## Layout

    src/          the world, the creatures, their networks, and the training code
    experiments/  one script per experiment; each writes to results/
    tests/        checks for every part
    results/      trained policies and the data behind the findings

`python3 tests/test_world.py` and its siblings run the checks. The tests under
`test_individual.py` need torch; the rest do not.

## Status

Work in progress.

## Simulation semantics and current limitations

Adult ticks increment age, release any finished nest occupancy, and settle maturation
before spending metabolic and crowding energy. The agent then acts, eats at its resulting
square, and is checked for starvation and old age. Eating can rescue an energy deficit;
it cannot prevent death from old age.

Initial food requests must fit the distinct squares inside the configured feeder regions.
Overlapping regions and walls reduce capacity. Infeasible requests raise an error before
partial placement; the simulation does not expand the food region or reduce the requested
budget silently. Limited-food regrowth stops at available capacity, and all replenishment
uses the configured feeder spread.

`UNIVERSE_25` uses 60 immediately replenished food items, four feeders, and spread
three. Its opt-in `feeder_regions_disjoint` placement rule keeps each 7×7 food region
fully inside the walls and prevents overlap. Every seed therefore has exactly 196 legal
food squares and the same 60/196 (30.6%) occupancy. Other configurations retain the
original unconstrained placement unless they explicitly opt in.

Placement samples four distinct cells of a randomly offset 7×7 tiling using the local
seeded world RNG. This constructive method avoids random rejection failures, reports
impossible placement capacity clearly, and restricts feeder centres to a common lattice;
it is not uniform sampling over all disjoint layouts. Existing perimeter insets, when
requested, are respected. Nest placement rules, rewards and network architecture are
unchanged. Food/nest overlap and distances still vary and should be measured.

With the previous unconstrained geometry, the 560 planned world seeds had capacities
of 60–196 squares: seed 3082's saturated region forced eaten food to reappear on the
same square, allowing repeated feeding without movement. Fixing capacity removes that
seed-dependent saturation confound without selecting for population growth or collapse.
The original request of 260 items remains infeasible at spread three. Replenishment can
still return food to an eaten square by chance, and does not guarantee unrestricted
individual access; record starvation, meals, stationary eating, and food/nest accessibility.

Regression checks cover all planned training seeds (episode seed = training seed * 1000
+ iteration), campaign/comparison seeds 0–2 and representation seeds 100–105. A guard
set of all five training seeds for 120 iterations covers 600 distinct world seeds.
Matched configurations start with identical feeders, food and nests for the same world
seed. The geometric capacity guarantee also holds for other seeds with this preset.

Campaign neighbour and clustering measurements describe the tick-start occupancy snapshot.
Clustering compares measured counts with the uniform-peer expectation at the same observed
positions, clipping neighbourhoods at walls. Population, birth, and death records describe
the end of the tick.

Parental care retains the existing timing: attendance uses the parent's position after
adult actions, while care-blocking and intrusion use tick-start crowding. Switching care
to a fully pre-action or post-action snapshot would change pup survival dynamics and needs
a scientific decision. Dead-individual trajectory handling and delayed reward credit
across learning windows also remain unresolved.
