import sys, json, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from config import UNIVERSE_25
from rl import train

mode = sys.argv[1]
seeds = [int(s) for s in sys.argv[2].split(',')]
PATH = str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'results' / 'rl_curves.json')

# The enclosure, with the population held low. These are the same conditions the founding
# policies are trained under, so this comparison now happens in the same pen as every other
# experiment. It previously ran in an earlier, simpler world with open edges and food
# scattered everywhere, which made it the one result that could not be set beside the rest.
cfg = UNIVERSE_25.variant('reward_comparison', n_ticks=500, n_initial_agents=40,
                          max_population=120)

out = json.load(open(PATH)) if os.path.exists(PATH) else []
for seed in seeds:
    policy, hist = train(cfg, reward_offspring=(mode == 'offspring'), iterations=100,
                         seed=seed, log_every=0, probe_every=10)
    probed = [h for h in hist if 'pup_seeking' in h]
    out = [x for x in out if not (x['mode'] == mode and x['seed'] == seed)]
    out.append({'mode': mode, 'seed': seed,
                'curve': [{'it': h['iteration'], 'pup': round(h['pup_seeking'], 3),
                           'food': round(h['food_seeking'], 3),
                           'conflict': round(h['care_vs_food'], 3),
                           'births': h['births']} for h in probed]})
    json.dump(out, open(PATH, 'w'), indent=1)
    last = probed[-1]
    print(f"{mode} seed {seed}: births {hist[0]['births']} -> {hist[-1]['births']} | "
          f"pup {last['pup_seeking']:.3f} | food {last['food_seeking']:.3f} | "
          f"conflict {last['care_vs_food']:+.3f}", flush=True)
