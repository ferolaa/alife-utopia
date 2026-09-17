"""What a trained policy represents, and which parts of it the behaviour needs.

Run this on one saved policy. It does four things.

First it gathers situations. Colonies are run in the pen with the policy driving them, and
every sense vector the population receives is kept. These are the situations the policy is
actually in, which is the only distribution in which a claim about what a unit does means
anything.

Then it asks three questions of the network.

  what each unit tracks   read the hidden layer in those situations
  what it needs to see    blind it to one sense at a time and re-measure the behaviour
  what it needs to have   silence hidden units, one at a time and then cumulatively

The interesting comparison is across reward functions. A policy trained on survival alone
has no reason to represent its young at all, so if these networks specialise, the two should
not need the same things. That comparison is what compare_representations.py does with the
files this writes.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from brain import SENSES  # noqa: E402
from config import UNIVERSE_25  # noqa: E402
from represent import (ablation, progressive_ablation, sense_ablation, sensitivity,
                       specialisation, tuning)  # noqa: E402
from rl import PolicyNet, run_episode  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# The pen the policies are trained in, so the situations sampled here are the ones that
# shaped them.
PEN = UNIVERSE_25.variant("represent", n_ticks=500, n_initial_agents=40, max_population=120)


def collect(policy, episodes: int = 6, seed: int = 100) -> np.ndarray:
    """Every sense vector the population receives over several colony runs."""
    sink: list = []
    for i in range(episodes):
        run_episode(PEN, policy, reward_offspring=True, seed=seed + i, senses_sink=sink)
    return np.concatenate(sink, axis=0)


def main(label: str):
    policy = PolicyNet()
    policy.load_state_dict(torch.load(ROOT / "results" / f"{label}.pt"))
    brain = policy.to_brain()

    situations = collect(policy)
    units = tuning(brain, situations)
    slopes = sensitivity(brain, situations)
    silenced = ablation(brain, trials=1200)
    blinded = sense_ablation(brain, trials=1200)
    curve = progressive_ablation(brain, trials=800)

    result = {
        "policy": label,
        "situations": int(situations.shape[0]),
        "sense_spread": {name: round(float(situations[:, i].std()), 4)
                         for i, name in enumerate(SENSES)},
        "units": units,
        "sensitivity": {name: [round(float(v), 4) for v in slopes[i]]
                        for i, name in enumerate(SENSES)},
        "ablation": silenced,
        "sense_ablation": blinded,
        "progressive": curve,
        "specialisation": specialisation(silenced),
    }
    json.dump(result, open(ROOT / "results" / f"represent_{label}.json", "w"), indent=1)

    print(f"{label}: {result['situations']} situations sampled")
    print(f"  intact: " + ", ".join(f"{k} {v:+.3f}" for k, v in silenced["intact"].items()))
    print()
    print("  what each unit responds to, and what silencing it costs")
    print("  unit  activity  tracks             r   largest weight      pup-seek  food-seek"
          "   conflict")
    for u, a in zip(units, silenced["units"]):
        strongest = SENSES[int(np.argmax(slopes[:, u["unit"]]))]
        print(f"  {u['unit']:4d} {u['activity']:9.3f}  {u['listens_to']:<14s} "
              f"{u['listens_strength']:+5.2f}   {strongest:<16s} "
              f"{a['change']['pup_seeking']:+9.3f} {a['change']['food_seeking']:+10.3f} "
              f"{a['change']['care_vs_food']:+10.3f}")
    print()
    # A probe only puts some of the senses in play. The food probe leaves every pup sense at
    # zero, so blinding the brain to them cannot change that number, and a zero there means
    # nothing. The conflict probe is the one where food and pup senses are both live, and it
    # is the column worth reading.
    print("  what it costs to take a sense away")
    print("  sense            pup-seek   food-seek    conflict")
    for s in blinded["senses"]:
        print(f"  {s['sense']:<14s} {s['change']['pup_seeking']:+9.3f} "
              f"{s['change']['food_seeking']:+11.3f} {s['change']['care_vs_food']:+11.3f}")
    print()
    print("  pup seeking as units are silenced, worst first")
    print("  " + "  ".join(f"{c['removed']}:{c['pup_seeking']:.3f}" for c in curve))


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["pen_policy_s0"]):
        main(name)
        print()
