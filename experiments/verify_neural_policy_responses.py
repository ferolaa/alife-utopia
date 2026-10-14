"""Recalculate saved same-state responses from frozen networks, without sampling."""
import argparse,sys,json,pathlib,numpy as np,torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.neural_policy_evaluation import load_policies
from brain import SENSES
assert list(SENSES[9:])==['pup_dx','pup_dy','pup_need']
torch.set_num_threads(1)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=pathlib.Path,default=ROOT/'results/neural_policy_v1')
parser.add_argument('--out',type=pathlib.Path,required=True)
args=parser.parse_args()
if args.out.exists():raise FileExistsError('Verification output must be NEW')
records=json.loads((args.data/'records.json').read_text());policies,_=load_policies(range(10));n=0
for seed in range(10):
 for cell in ['initial','survival_normal','offspring_normal','survival_masked','offspring_masked']:
  rows=[a for r in records if r['training_seed']==seed and r['cell']==cell for a in r['input_audits']]
  policy=policies[(seed,cell.split('_')[0])]
  for i in range(0,len(rows),256):
   part=rows[i:i+256];raw=torch.tensor([a['raw_input'] for a in part]);used=raw.clone()
   if cell.endswith('masked'):used[:,9:]=0
   with torch.no_grad():
    normal=policy(raw);applied=policy(used)
    p=torch.softmax(normal,-1);q=torch.softmax(applied,-1)
    kl=(p*(torch.log_softmax(normal,-1)-torch.log_softmax(applied,-1))).sum(-1).numpy()
   assert np.allclose(p.numpy(),[a['normal_probabilities'] for a in part],rtol=1e-5,atol=2e-6)
   assert np.allclose(q.numpy(),[a['applied_probabilities'] for a in part],rtol=1e-5,atol=2e-6)
   assert np.allclose(kl,[a['normal_to_applied_kl'] for a in part],rtol=1e-5,atol=2e-6)
   n+=len(part)
print(n,'stored policy responses independently recalculated from frozen parameters; only channels9–11 masked.')
args.out.write_text(json.dumps({'responses_recalculated':n,'probability_kl_tolerance_absolute':2e-6,'probability_kl_tolerance_relative':1e-5,'input_channels_verified':['pup_dx','pup_dy','pup_need']}))
