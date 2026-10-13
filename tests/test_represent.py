"""Checks on the representation analysis.

These use brains with weights chosen by hand rather than trained ones, because the point is
to check that the measurements say what they claim. A trained brain would tell us nothing
here: if the answer came out strange we would not know whether the analysis was wrong or the
brain was.
"""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain import ACTIONS, SENSES, Brain, make_senses  # noqa: E402
from represent import (ablation, progressive_ablation, sense_ablation, sensitivity,
                       specialisation, tuning)  # noqa: E402

SITUATIONS = np.array([
    make_senses(food_dx=1.0, food_closeness=0.8, energy=0.5),
    make_senses(food_dx=-1.0, food_closeness=0.2, energy=0.9),
    make_senses(pup_dx=1.0, pup_need=0.7, energy=0.4),
    make_senses(crowding=1.0, energy=0.1),
    make_senses(nest_dy=1.0, nest_closeness=0.5, energy=0.7),
])


def a_brain(seed=0):
    return Brain.random(random.Random(seed))


def test_tuning_describes_every_unit():
    units = tuning(a_brain(), SITUATIONS)
    assert len(units) == Brain.N_HIDDEN
    for i, u in enumerate(units):
        assert u["unit"] == i
        assert set(u["senses"]) == set(SENSES)
        assert set(u["votes_for"]) == set(ACTIONS)
        assert u["listens_to"] in SENSES
        assert -1.0 <= u["listens_strength"] <= 1.0
        assert u["low"] <= u["mean"] <= u["high"]


def test_tuning_rejects_the_wrong_shape():
    try:
        tuning(a_brain(), np.zeros((4, 3)))
    except ValueError:
        return
    raise AssertionError("expected a ValueError for a matrix with the wrong width")


def test_a_unit_that_only_watches_one_sense_is_reported_as_watching_it():
    """A brain wired by hand: unit zero reads energy and nothing else.

    Everything else in the first layer is left at zero, so the other units sit flat and only
    unit zero has anything to correlate with.
    """
    weights = np.zeros(Brain.N_WEIGHTS)
    energy = SENSES.index("energy")
    weights[energy * Brain.N_HIDDEN + 0] = 2.0     # energy into unit zero
    brain = Brain(weights)

    units = tuning(brain, SITUATIONS)
    assert units[0]["listens_to"] == "energy"
    assert units[0]["listens_strength"] > 0.9
    assert units[0]["activity"] > 0.0
    # Every other unit is constant, so nothing correlates with anything.
    for u in units[1:]:
        assert u["activity"] == 0.0
        assert all(v == 0.0 for v in u["senses"].values())


def test_a_flat_unit_votes_for_nothing():
    """Outgoing weight is not influence. A unit that never moves changes no decision."""
    weights = np.zeros(Brain.N_WEIGHTS)
    # Unit zero has a large weight onto an action, but nothing feeding into it.
    second_layer = Brain.N_INPUTS * Brain.N_HIDDEN + Brain.N_HIDDEN
    weights[second_layer + 0] = 5.0
    units = tuning(Brain(weights), SITUATIONS)
    assert all(v == 0.0 for v in units[0]["votes_for"].values())


def test_sensitivity_has_a_row_per_sense_and_a_column_per_unit():
    slopes = sensitivity(a_brain(), SITUATIONS)
    assert slopes.shape == (Brain.N_INPUTS, Brain.N_HIDDEN)
    assert np.all(slopes >= 0.0)


def test_a_saturated_unit_is_not_listening():
    """Huge weights push tanh flat, so the unit stops responding to anything."""
    small = np.zeros(Brain.N_WEIGHTS)
    energy = SENSES.index("energy")
    small[energy * Brain.N_HIDDEN + 0] = 1.0
    huge = small.copy()
    huge[:Brain.N_INPUTS * Brain.N_HIDDEN] = 0.0
    huge[energy * Brain.N_HIDDEN + 0] = 1.0
    huge[Brain.N_INPUTS * Brain.N_HIDDEN + 0] = 30.0      # a bias that saturates the unit

    calm = sensitivity(Brain(small), SITUATIONS)[energy, 0]
    saturated = sensitivity(Brain(huge), SITUATIONS)[energy, 0]
    assert saturated < calm / 10


def test_ablation_covers_every_unit_and_is_measured_against_the_intact_brain():
    result = ablation(a_brain(1), trials=60)
    assert len(result["units"]) == Brain.N_HIDDEN
    keys = set(result["intact"])
    for entry in result["units"]:
        assert set(entry["change"]) == keys
        for k in keys:
            assert abs(entry["scores"][k] - result["intact"][k] - entry["change"][k]) < 1e-9


def test_sense_ablation_covers_every_sense():
    result = sense_ablation(a_brain(2), trials=60)
    assert [s["sense"] for s in result["senses"]] == list(SENSES)


def test_blinding_a_sense_the_probe_never_uses_changes_nothing():
    """The food probe leaves every pup sense at zero, so removing one cannot matter.

    This is worth pinning down because it is the trap in reading these tables. A zero in the
    wrong column is a fact about the probe, not about the network.
    """
    result = sense_ablation(a_brain(3), trials=200)
    for name in ("pup_dx", "pup_dy", "pup_need"):
        entry = next(s for s in result["senses"] if s["sense"] == name)
        assert entry["change"]["food_seeking"] == 0.0


def test_progressive_ablation_removes_one_more_unit_each_step():
    curve = progressive_ablation(a_brain(4), trials=60)
    assert len(curve) == Brain.N_HIDDEN
    for step, point in enumerate(curve):
        assert point["removed"] == step
        assert len(point["units"]) == step
        assert len(set(point["units"])) == step      # never the same unit twice


def test_specialisation_is_one_when_a_single_unit_carries_everything():
    made_up = {"units": [{"change": {"pup_seeking": -0.4}}]
                        + [{"change": {"pup_seeking": 0.0}} for _ in range(11)]}
    assert abs(specialisation(made_up) - 1.0) < 1e-9


def test_specialisation_is_a_twelfth_when_every_unit_matters_equally():
    made_up = {"units": [{"change": {"pup_seeking": -0.1}} for _ in range(12)]}
    assert abs(specialisation(made_up) - 1 / 12) < 1e-9


def test_specialisation_is_zero_when_nothing_is_lost():
    made_up = {"units": [{"change": {"pup_seeking": 0.05}} for _ in range(12)]}
    assert specialisation(made_up) == 0.0


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
                print(f"PASS {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
