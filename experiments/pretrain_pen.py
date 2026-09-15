"""Phase one: raise a competent population inside the enclosure, at low density.

Calhoun began with normal, healthy mice. He was not asking whether mice can learn to
forage and raise young; he was asking whether mice that already can will stop.

So the agents here are trained in the enclosure itself, with everything identical to the
experiment proper except one number: the population is held low. They learn to find the
feeders, claim a nest and stay near their young, but they never experience a crowd.

Holding the geometry fixed and varying only the population cap means that when the cap is
lifted in phase two, density is the one thing that changed. Anything that happens to their
behaviour afterwards can be laid at its door.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from config import UNIVERSE_25  # noqa: E402
from evaluate import care_under_conflict, food_seeking_score, pup_seeking_score  # noqa: E402
from rl import train  # noqa: E402

# The enclosure, with the population kept small. Short episodes, since the policy is shared
# across every creature alive and one episode already supplies thousands of decisions.
PRETRAIN = UNIVERSE_25.variant(
    "pretrain",
    n_ticks=500,
    n_initial_agents=40,
    max_population=120,      # the only thing that separates this from the experiment
)

if __name__ == "__main__":
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    policy, history = train(
        PRETRAIN, reward_offspring=True, iterations=iterations,
        seed=seed, log_every=20, probe_every=20,
    )

    out = Path(__file__).resolve().parents[1] / "results" / f"pen_policy_s{seed}.pt"
    torch.save(policy.state_dict(), out)

    brain = policy.to_brain()
    print()
    print(f"saved {out.name}")
    print(f"  food seeking  {food_seeking_score(brain, trials=800):.3f}   (chance 0.25)")
    print(f"  pup seeking   {pup_seeking_score(brain, trials=800):.3f}")
    print(f"  pup vs food   {care_under_conflict(brain, trials=800)['preference']:+.3f}")
    print(f"  final episode: population {history[-1]['final_population']}, "
          f"births {history[-1]['births']}")
