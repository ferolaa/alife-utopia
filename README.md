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
