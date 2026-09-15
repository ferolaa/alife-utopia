import sys, json, os
sys.path.insert(0, 'src')
from config import DEFAULT
from rl import train

mode = sys.argv[1]
seeds = [int(s) for s in sys.argv[2].split(',')]
PATH = 'results/rl_curves.json'

cfg = DEFAULT.variant('rl', n_ticks=400, n_initial_agents=80, max_population=400,
                      ageing_enabled=True, food_unlimited=True, n_food=300,
                      nests_enabled=True, n_nests=40, parental_care_enabled=True,
                      crowding_cost_enabled=True)

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
