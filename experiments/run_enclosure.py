"""Phase two: the experiment.

A small group of creatures, already competent at feeding themselves and raising young, is
placed in the enclosure and left there. The reward they are trained on does not change. The
enclosure does not change. The population grows.

The question is Calhoun's: does behaviour that was working break down as the place fills up?
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
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    policy_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    policy = PolicyNet()
    policy.load_state_dict(torch.load(ROOT / "results" / f"pen_policy_s{policy_seed}.pt"))
    optimiser = torch.optim.Adam(policy.parameters(), lr=0.003)

    cfg = UNIVERSE_25.variant("enclosure", n_ticks=ticks, seed=seed)
    print(f"founding with {cfg.n_initial_agents} creatures in a {cfg.width}x{cfg.height} pen, "
          f"{cfg.n_nests} nests, {cfg.n_feeders} feeders, cap {cfg.max_population}")
    world, history = run_enclosure(cfg, policy, optimiser, seed=seed)

    keep = ("tick", "population", "births", "deaths", "died_neglected", "died_starving",
            "died_old", "nests_free", "mean_neighbours", "clustering",
            "pup_seeking", "food_seeking")
    out = ROOT / "results" / f"enclosure_s{seed}.json"
    json.dump([{k: h[k] for k in keep if k in h} for h in history], open(out, "w"))
    peak = max(h["population"] for h in history)
    peak_at = next(h["tick"] for h in history if h["population"] == peak)
    print()
    print(f"peak {peak} at tick {peak_at} | final {history[-1]['population']} | "
          f"ticks run {len(history)}")
