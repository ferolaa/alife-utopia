"""Checks for saving and reusing an evolved population."""

import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain import Brain  # noqa: E402
from population import draw_founders, load_brains, save_brains  # noqa: E402


def _some_brains(n=5):
    return [Brain.random(random.Random(i)) for i in range(n)]


def test_saving_then_loading_gives_back_the_same_weights():
    brains = _some_brains()
    with tempfile.TemporaryDirectory() as tmp:
        path = save_brains(brains, Path(tmp) / "p.npz")
        loaded = load_brains(path)
    assert len(loaded) == len(brains)
    for before, after in zip(brains, loaded):
        assert np.allclose(before.weights, after.weights)


def test_loading_a_missing_file_says_so_clearly():
    try:
        load_brains(Path("/nonexistent/nothing.npz"))
    except FileNotFoundError:
        return
    raise AssertionError("expected a FileNotFoundError")


def test_brains_of_the_wrong_size_are_refused():
    # A population saved before the brain gained inputs must not load silently.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old.npz"
        np.savez_compressed(path, weights=np.zeros((3, Brain.N_WEIGHTS - 1)))
        try:
            load_brains(path)
        except ValueError:
            return
    raise AssertionError("expected a ValueError for out of date brains")


def test_founders_are_drawn_from_the_saved_population():
    brains = _some_brains(3)
    drawn = draw_founders(brains, 10, random.Random(0))
    assert len(drawn) == 10                       # more founders than saved brains is fine
    known = [b.weights.tobytes() for b in brains]
    assert all(d.weights.tobytes() in known for d in drawn)


def test_drawn_founders_are_copies_not_shared():
    brains = _some_brains(2)
    drawn = draw_founders(brains, 4, random.Random(1))
    drawn[0].weights[0] = 12345.0                 # changing a founder
    assert all(b.weights[0] != 12345.0 for b in brains)   # must not touch the saved stock


def test_drawing_from_nothing_is_refused():
    try:
        draw_founders([], 3, random.Random(0))
    except ValueError:
        return
    raise AssertionError("expected a ValueError")


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
