"""The control arm: one policy driving every creature in the pen.

Each run pairs a pretrained policy with a world seed of the same number, so the three runs
are independent replicates rather than three views of the same brain. The individual arm
uses the identical pairing, which is what makes the two arms comparable.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from config import UNIVERSE_25  # noqa: E402
from rl import PolicyNet, run_enclosure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")]
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 6000

    for seed in seeds:
        policy = PolicyNet()
        policy.load_state_dict(torch.load(ROOT / "results" / f"pen_policy_s{seed}.pt"))
        optimiser = torch.optim.Adam(policy.parameters(), lr=0.003)

        cfg = UNIVERSE_25.variant(f"shared_s{seed}", n_ticks=ticks, seed=seed)
        world, history = run_enclosure(cfg, policy, optimiser, seed=seed,
                                       probe_every=500, verbose=False)

        keep = ("tick", "population", "births", "deaths", "died_neglected",
                "died_starving", "died_old", "nests_free", "mean_neighbours",
                "clustering", "pup_seeking", "food_seeking")
        out = ROOT / "results" / f"campaign_shared_s{seed}.json"
        json.dump([{k: h[k] for k in keep if k in h} for h in history], open(out, "w"))

        peak = max(h["population"] for h in history)
        tot = lambda k: sum(h[k] for h in history)
        probes = [h["pup_seeking"] for h in history if "pup_seeking" in h]
        print(f"seed {seed}: peak {peak}, final {history[-1]['population']}, "
              f"neglect {tot('died_neglected')}, starved {tot('died_starving')}, "
              f"pup-seeking {probes[0]:.3f} -> {probes[-1]:.3f}", flush=True)
