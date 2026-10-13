"""Check that the pretrained policies are usable as replicates.

Three separately trained policies only count as replicates if they are comparably
competent. If one were markedly better than the others, differences between runs would be
telling us which brain got lucky rather than anything about the conditions being compared.

This measures each one two ways: the behaviour probes, which describe a policy in
isolation, and an actual colony run, which describes what it does. The second matters more.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from config import UNIVERSE_25  # noqa: E402
from evaluate import care_under_conflict, food_seeking_score, pup_seeking_score  # noqa: E402
from rl import PolicyNet, run_enclosure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2").split(",")]
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 2500

    rows = []
    for s in seeds:
        policy = PolicyNet()
        policy.load_state_dict(torch.load(ROOT / "results" / f"pen_policy_s{s}.pt"))
        brain = policy.to_brain()

        optimiser = torch.optim.Adam(policy.parameters(), lr=0.003)
        cfg = UNIVERSE_25.variant(f"compare_s{s}", n_ticks=ticks, seed=0)
        world, history = run_enclosure(cfg, policy, optimiser, seed=0,
                                       probe_every=0, verbose=False)
        late = history[-500:]
        rows.append({
            "policy": s,
            "food": round(food_seeking_score(brain, trials=800), 3),
            "pup": round(pup_seeking_score(brain, trials=800), 3),
            "conflict": round(care_under_conflict(brain, trials=800)["preference"], 3),
            "peak": max(h["population"] for h in history),
            "late_pop": round(sum(h["population"] for h in late) / len(late), 1),
            "births": sum(h["births"] for h in history),
            "neglect": sum(h["died_neglected"] for h in history),
        })
        print(f"policy {s}: peak {rows[-1]['peak']}, late population {rows[-1]['late_pop']}, "
              f"births {rows[-1]['births']}", flush=True)

    json.dump(rows, open(ROOT / "results" / "policy_comparison.json", "w"), indent=1)

    print()
    print("policy  food-seek  pup-seek  conflict   peak  late_pop  births  neglect")
    for r in rows:
        print(f"{r['policy']:6d} {r['food']:10.3f} {r['pup']:9.3f} {r['conflict']:+9.3f} "
              f"{r['peak']:6d} {r['late_pop']:9.1f} {r['births']:7d} {r['neglect']:8d}")
