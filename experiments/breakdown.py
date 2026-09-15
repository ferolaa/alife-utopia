"""Do the density-activated mechanisms break a competent population?

Three additions, each a thing Calhoun described and each dormant until the pen fills up:

  blocked    a crowd gets between parent and pup, so being near stops being enough
  intrusion  a crowded nest harms the pup whatever its parent does
  damage     a pup that survives neglect grows into a parent who cannot tend its own

Because all three only bite at density, the policy trained at low density has never met
them, and nothing about them makes the low-density world it learned in any different. The
population grows into them.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from config import UNIVERSE_25  # noqa: E402
from rl import PolicyNet, run_enclosure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = {
    "none": {},
    "blocked": dict(crowding_blocks_care=True),
    "intrusion": dict(nest_intrusion_enabled=True),
    "damage": dict(developmental_damage_enabled=True),
    "all": dict(crowding_blocks_care=True, nest_intrusion_enabled=True,
                developmental_damage_enabled=True),
}

if __name__ == "__main__":
    name = sys.argv[1]
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    policy = PolicyNet()
    policy.load_state_dict(torch.load(ROOT / "results" / "pen_policy_s0.pt"))
    optimiser = torch.optim.Adam(policy.parameters(), lr=0.003)

    cfg = UNIVERSE_25.variant(name, n_ticks=ticks, seed=seed, **CONDITIONS[name])
    print(f"condition '{name}': {CONDITIONS[name] or 'baseline'}")
    world, history = run_enclosure(cfg, policy, optimiser, seed=seed, probe_every=1000)

    keep = ("tick", "population", "births", "deaths", "died_neglected", "died_starving",
            "died_old", "nests_free", "mean_neighbours", "clustering", "pup_seeking")
    out = ROOT / "results" / f"breakdown_{name}_s{seed}.json"
    json.dump([{k: h[k] for k in keep if k in h} for h in history], open(out, "w"))

    peak = max(h["population"] for h in history)
    peak_at = next(h["tick"] for h in history if h["population"] == peak)
    tot = lambda k: sum(h[k] for h in history)
    print()
    print(f"  peak {peak} at tick {peak_at} | final {history[-1]['population']} | "
          f"ran {len(history)} ticks")
    print(f"  deaths: neglect {tot('died_neglected')}, starvation {tot('died_starving')}, "
          f"old age {tot('died_old')}")
