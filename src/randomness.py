"""Isolate seeded CPU Torch runs from unrelated global RNG activity.

Python world RNGs and NumPy brain generators are already local. Torch distributions
use the global CPU generator, so save/restore it around each synchronous run.
This context is not intended for concurrent threads or CUDA execution.
"""

from functools import wraps
from inspect import signature

import torch


def seeded_run(fn):
    sig = signature(fn)

    @wraps(fn)
    def run(*args, **kwargs):
        arguments = sig.bind(*args, **kwargs)
        arguments.apply_defaults()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(arguments.arguments["seed"])
            return fn(*args, **kwargs)

    return run
