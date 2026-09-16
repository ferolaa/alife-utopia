"""Checks for a population where every creature carries its own network."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain import Brain, make_senses  # noqa: E402
from individual import Population  # noqa: E402
from rl import PolicyNet  # noqa: E402

SENSES = make_senses(food_dx=0.5, pup_need=0.3, energy=0.7)


def _filled(n=4, capacity=8):
    pop = Population(capacity=capacity)
    policy = PolicyNet()
    slots = [pop.claim() for _ in range(n)]
    for s in slots:
        pop.set_from_policy(s, policy)
    return pop, policy, slots


def test_slots_are_handed_out_and_taken_back():
    pop = Population(capacity=3)
    a, b, c = pop.claim(), pop.claim(), pop.claim()
    assert pop.in_use == 3
    assert pop.claim() is None            # full
    pop.release(b)
    assert pop.in_use == 2
    assert pop.claim() == b               # the freed slot comes back


def test_a_copied_policy_behaves_exactly_like_the_original():
    pop, policy, slots = _filled(3)
    senses = torch.tensor([SENSES] * 3, dtype=torch.float32)
    batched = pop.action_scores(slots, senses).detach().numpy()
    single = policy(torch.tensor([SENSES], dtype=torch.float32)).detach().numpy()[0]
    assert np.allclose(batched[0], single, atol=1e-5)


def test_each_creature_is_run_through_its_own_weights():
    """The whole point: different slots must give different answers.

    If the batched einsum indexed wrongly, every creature would silently share one network
    and the population could never differentiate, which is the one thing this class exists
    to make possible.
    """
    pop, _, slots = _filled(3)
    with torch.no_grad():
        pop.W1[slots[1]] += 1.0
    senses = torch.tensor([SENSES] * 3, dtype=torch.float32)
    out = pop.action_scores(slots, senses).detach().numpy()
    assert np.allclose(out[0], out[2], atol=1e-6)
    assert not np.allclose(out[0], out[1], atol=1e-3)


def test_order_of_slots_is_respected():
    pop, _, slots = _filled(3)
    with torch.no_grad():
        pop.W1[slots[0]] += 1.0
    senses = torch.tensor([SENSES] * 3, dtype=torch.float32)
    forward = pop.action_scores(slots, senses).detach().numpy()
    backward = pop.action_scores(slots[::-1], senses).detach().numpy()
    assert np.allclose(forward[0], backward[2], atol=1e-6)


def test_inheritance_copies_the_parents_weights():
    pop, _, slots = _filled(2)
    with torch.no_grad():
        pop.W1[slots[0]] += 0.7
    pop.copy_slot(slots[0], slots[1])
    senses = torch.tensor([SENSES] * 2, dtype=torch.float32)
    out = pop.action_scores(slots, senses).detach().numpy()
    assert np.allclose(out[0], out[1], atol=1e-6)


def test_a_copy_is_not_a_shared_reference():
    # A child must not go on sharing memory with its parent, or training one would train
    # the other and lineages could never diverge.
    pop, _, slots = _filled(2)
    pop.copy_slot(slots[0], slots[1])
    with torch.no_grad():
        pop.W1[slots[0]] += 5.0
    senses = torch.tensor([SENSES] * 2, dtype=torch.float32)
    out = pop.action_scores(slots, senses).detach().numpy()
    assert not np.allclose(out[0], out[1], atol=1e-3)


def test_spread_is_zero_when_everyone_is_identical():
    pop, _, slots = _filled(5)
    assert pop.spread(slots) < 1e-6


def test_spread_grows_as_creatures_differ():
    pop, _, slots = _filled(5)
    before = pop.spread(slots)
    with torch.no_grad():
        pop.W1[slots[0]] += 0.5
    middling = pop.spread(slots)
    with torch.no_grad():
        pop.W1[slots[1]] -= 1.5
    after = pop.spread(slots)
    assert before < middling < after


def test_spread_needs_at_least_two_creatures():
    pop, _, slots = _filled(1)
    assert pop.spread(slots) == 0.0
    assert pop.spread([]) == 0.0


def test_weights_round_trip_to_a_plain_brain():
    pop, policy, slots = _filled(1)
    brain = pop.brain_of(slots[0])
    assert brain.weights.shape == (Brain.N_WEIGHTS,)
    expected = policy(torch.tensor([SENSES], dtype=torch.float32)).detach().numpy()[0]
    assert np.allclose(brain.action_scores(SENSES), expected, atol=1e-5)



def test_a_founder_is_a_copy_of_the_pretrained_policy():
    pop = Population(capacity=4)
    policy = PolicyNet()
    slot = pop.founder(policy)
    expected = policy(torch.tensor([SENSES], dtype=torch.float32)).detach().numpy()[0]
    assert np.allclose(pop.brain_of(slot).action_scores(SENSES), expected, atol=1e-5)


def test_a_newborn_starts_knowing_what_its_parent_knew():
    pop = Population(capacity=4)
    parent = pop.founder(PolicyNet())
    with torch.no_grad():
        pop.W2[parent] += 0.4          # the parent learnt something in its life
    child = pop.inherit(parent)
    assert np.allclose(pop.brain_of(parent).weights, pop.brain_of(child).weights, atol=1e-6)
    assert pop.spread([parent, child]) < 1e-6


def test_a_child_that_learns_drifts_from_its_parent():
    pop = Population(capacity=4)
    parent = pop.founder(PolicyNet())
    child = pop.inherit(parent)
    before = pop.spread([parent, child])
    with torch.no_grad():
        pop.W1[child] += 0.3           # a life of its own
    assert before == 0.0
    assert pop.spread([parent, child]) > 0.1


def test_inheritance_is_noiseless_by_default():
    """No mutation unless asked for.

    This keeps the experiment interpretable: with noise off, any difference between two
    creatures was caused by something that happened to one of them, not by randomness we
    sprinkled on at birth.
    """
    pop = Population(capacity=4)
    parent = pop.founder(PolicyNet())
    child = pop.inherit(parent)
    assert np.array_equal(pop.brain_of(parent).weights, pop.brain_of(child).weights)


def test_mutation_can_be_switched_on():
    import random as _random
    pop = Population(capacity=4)
    parent = pop.founder(PolicyNet())
    child = pop.inherit(parent, mutation_std=0.1, rng=_random.Random(0))
    diff = np.abs(pop.brain_of(parent).weights - pop.brain_of(child).weights)
    assert diff.mean() > 0
    assert diff.mean() < 0.5           # still recognisably its parent


def test_a_full_population_refuses_to_add_more():
    pop = Population(capacity=2)
    parent = pop.founder(PolicyNet())
    assert pop.inherit(parent) is not None     # fills the second slot
    assert pop.inherit(parent) is None         # no room
    assert pop.founder(PolicyNet()) is None


def test_released_slots_come_back_for_newborns():
    pop = Population(capacity=2)
    parent = pop.founder(PolicyNet())
    child = pop.inherit(parent)
    pop.release(child)                          # the child dies
    again = pop.inherit(parent)                 # another is born
    assert again == child                       # and reuses the slot
    assert pop.in_use == 2


def test_a_reused_slot_does_not_keep_the_dead_creatures_weights():
    # Slot reuse is where a stale-state bug would hide: a newborn inheriting whatever the
    # previous occupant had learnt, rather than what its own parent knows.
    pop = Population(capacity=2)
    first = pop.founder(PolicyNet())
    ghost = pop.inherit(first)
    with torch.no_grad():
        pop.W1[ghost] += 9.0                    # the dead creature was very unusual
    pop.release(ghost)
    newborn = pop.inherit(first)
    assert newborn == ghost                     # same slot
    assert np.allclose(pop.brain_of(newborn).weights, pop.brain_of(first).weights, atol=1e-6)


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
