import sys, json, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from config import DEFAULT
from rl import train
from evaluate import food_seeking_score, pup_seeking_score, care_under_conflict, CHANCE_LEVEL

mode = sys.argv[1]                      # 'survival' or 'offspring'
seed = int(sys.argv[2])
iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 100
PATH = 'results/rl_results.json'

cfg = DEFAULT.variant('rl', n_ticks=400, n_initial_agents=80, max_population=400,
                      ageing_enabled=True, food_unlimited=True, n_food=300,
                      nests_enabled=True, n_nests=40, parental_care_enabled=True,
                      crowding_cost_enabled=True)

policy, hist = train(cfg, reward_offspring=(mode == 'offspring'),
                     iterations=iterations, seed=seed, log_every=20)

brain = policy.to_brain()
probes = {
    'food_seeking': round(food_seeking_score(brain, trials=800), 3),
    'pup_seeking': round(pup_seeking_score(brain, trials=800), 3),
    'care_vs_food': round(care_under_conflict(brain, trials=800)['preference'], 3),
}
torch.save(policy.state_dict(), f'results/policy_{mode}_s{seed}.pt')

out = json.load(open(PATH)) if os.path.exists(PATH) else []
out = [x for x in out if not (x['mode'] == mode and x['seed'] == seed)]
out.append({'mode': mode, 'seed': seed, 'iterations': iterations, 'probes': probes,
            'final_population': hist[-1]['final_population'],
            'births_first': hist[0]['births'], 'births_last': hist[-1]['births'],
            'return_first': round(hist[0]['mean_return'], 3),
            'return_last': round(hist[-1]['mean_return'], 3)})
json.dump(out, open(PATH, 'w'), indent=1)

print()
print(f'{mode} seed {seed}:')
print(f"  births: {hist[0]['births']} -> {hist[-1]['births']}   mean return: {hist[0]['mean_return']:.2f} -> {hist[-1]['mean_return']:.2f}")
print(f"  food-seeking {probes['food_seeking']} | pup-seeking {probes['pup_seeking']} | pup-vs-food {probes['care_vs_food']:+.3f}   (chance {CHANCE_LEVEL})")
