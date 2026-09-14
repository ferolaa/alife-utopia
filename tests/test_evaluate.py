"""Checks on the behaviour probes themselves."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain import Brain  # noqa: E402
from evaluate import CHANCE_LEVEL, crowding_response, food_seeking_score, summarise_population  # noqa: E402


def test_score_is_a_fraction():
    b = Brain.random(random.Random(0))
    s = food_seeking_score(b, trials=50)
    assert 0.0 <= s <= 1.0


def test_random_brains_score_near_chance():
    # Averaged over many random brains, there should be no directional preference.
    brains = [Brain.random(random.Random(i)) for i in range(25)]
    mean = sum(food_seeking_score(b, trials=100, seed=i) for i, b in enumerate(brains)) / 25
    assert abs(mean - CHANCE_LEVEL) < 0.1


def test_crowding_response_reports_both_situations():
    b = Brain.random(random.Random(3))
    out = crowding_response(b, trials=50)
    assert set(out["alone"]) == {"reproduce", "move", "stay"}
    assert abs(sum(out["alone"].values()) - 1.0) < 1e-9
    assert abs(sum(out["crowded"].values()) - 1.0) < 1e-9


def test_reproduce_change_is_a_difference():
    b = Brain.random(random.Random(4))
    out = crowding_response(b, trials=50)
    expected = out["crowded"]["reproduce"] - out["alone"]["reproduce"]
    assert abs(out["reproduce_change"] - expected) < 1e-9


def test_summarise_handles_an_empty_population():
    assert summarise_population([])["n"] == 0


def test_summarise_averages_a_population():
    brains = [Brain.random(random.Random(i)) for i in range(5)]
    out = summarise_population(brains)
    assert out["n"] == 5
    assert out["food_seeking_min"] <= out["food_seeking"] <= out["food_seeking_max"]


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
