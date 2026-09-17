"""Sanity checks for the evolved brain."""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain import ACTIONS, SENSES, Brain, make_senses  # noqa: E402

# Built by name rather than as a bare tuple. A positional tuple breaks silently whenever
# the sense list changes, which is exactly what happened when the brain gained inputs for
# nests and pups.
SOME_SENSES = make_senses(food_dx=0.5, food_dy=-0.2, food_closeness=0.8, energy=0.6, crowding=0.1)


def test_random_brain_has_the_expected_number_of_weights():
    b = Brain.random(random.Random(0))
    assert b.weights.shape == (Brain.N_WEIGHTS,)
    assert Brain.N_WEIGHTS == (
        Brain.N_INPUTS * Brain.N_HIDDEN + Brain.N_HIDDEN
        + Brain.N_HIDDEN * Brain.N_OUTPUTS + Brain.N_OUTPUTS
    )


def test_make_senses_rejects_unknown_names():
    try:
        make_senses(not_a_sense=1.0)
    except ValueError:
        return
    raise AssertionError("expected a ValueError for an unknown sense name")


def test_make_senses_matches_the_sense_list():
    assert len(make_senses()) == len(SENSES)
    assert all(v == 0.0 for v in make_senses())


def test_wrong_weight_count_is_rejected():
    try:
        Brain(np.zeros(3))
    except ValueError:
        return
    raise AssertionError("expected a ValueError for the wrong number of weights")


def test_one_score_per_action():
    b = Brain.random(random.Random(1))
    assert b.action_scores(SOME_SENSES).shape == (len(ACTIONS),)


def test_senses_and_inputs_line_up():
    assert Brain.N_INPUTS == len(SENSES)
    assert Brain.N_OUTPUTS == len(ACTIONS)


def test_decide_returns_a_real_action():
    b = Brain.random(random.Random(2))
    rng = random.Random(3)
    for _ in range(50):
        assert b.decide(SOME_SENSES, rng) in ACTIONS


def test_thinking_is_deterministic_for_the_same_senses():
    b = Brain.random(random.Random(4))
    first = b.action_scores(SOME_SENSES)
    second = b.action_scores(SOME_SENSES)
    assert np.allclose(first, second)


def test_mutation_changes_the_weights_a_little():
    b = Brain.random(random.Random(5))
    child = b.mutated_copy(0.1, random.Random(6))
    diff = np.abs(child.weights - b.weights)
    assert diff.mean() > 0            # something changed
    assert diff.mean() < 0.5          # but it is still recognisably the parent


def test_zero_mutation_produces_an_identical_child():
    b = Brain.random(random.Random(7))
    child = b.mutated_copy(0.0, random.Random(8))
    assert np.allclose(child.weights, b.weights)


def test_parent_is_not_modified_by_having_a_child():
    b = Brain.random(random.Random(9))
    before = b.weights.copy()
    b.mutated_copy(0.3, random.Random(10))
    assert np.allclose(b.weights, before)


def test_hidden_activations_take_one_or_many_sense_vectors():
    b = Brain.random(random.Random(13))
    stack = np.array([SOME_SENSES, make_senses(energy=1.0), make_senses(pup_need=0.5)])
    assert b.hidden_activations(SOME_SENSES).shape == (Brain.N_HIDDEN,)
    assert b.hidden_activations(stack).shape == (len(stack), Brain.N_HIDDEN)
    assert np.allclose(b.hidden_activations(stack)[0], b.hidden_activations(SOME_SENSES))


def test_hidden_activations_stay_inside_the_tanh_range():
    b = Brain.random(random.Random(14))
    huge = np.full(Brain.N_INPUTS, 50.0)
    assert np.all(np.abs(b.hidden_activations(huge)) <= 1.0)


def test_silencing_a_unit_removes_exactly_that_unit():
    b = Brain.random(random.Random(15))
    hidden = b.hidden_activations(SOME_SENSES)
    for unit in range(Brain.N_HIDDEN):
        quiet = b.without_unit(unit)
        # The hidden layer itself is untouched. Only what it feeds into changes.
        assert np.allclose(quiet.hidden_activations(SOME_SENSES), hidden)
        # And the result is what you get by setting that one activation to zero by hand.
        by_hand = hidden.copy()
        by_hand[unit] = 0.0
        assert np.allclose(quiet.action_scores(SOME_SENSES), by_hand @ b._W2 + b._b2)


def test_silencing_leaves_the_original_brain_alone():
    b = Brain.random(random.Random(16))
    before = b.weights.copy()
    b.without_unit(0)
    assert np.allclose(b.weights, before)


def test_silencing_a_unit_that_does_not_exist_is_rejected():
    b = Brain.random(random.Random(17))
    for bad in (-1, Brain.N_HIDDEN):
        try:
            b.without_unit(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected a ValueError for hidden unit {bad}")


def test_low_temperature_concentrates_on_the_best_action():
    b = Brain.random(random.Random(11))
    rng = random.Random(12)
    best = ACTIONS[int(np.argmax(b.action_scores(SOME_SENSES)))]
    picks = [b.decide(SOME_SENSES, rng, temperature=0.01) for _ in range(100)]
    assert picks.count(best) > 90


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
