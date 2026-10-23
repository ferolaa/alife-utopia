"""Review-only CLI. No scientific pilot launch option is implemented here."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from v2.care_task import CareControl,CareTask,evaluate_care,panel_fingerprint
from v2.care_experiment import CareExperiment,protocol
from v2.experiment import provenance
from v2.learning import LearnerConfig
from v2.models import HandDesigned,Policy,Uniform


def review(output,smoke=False):
    output=Path(output).resolve()
    root=Path(__file__).resolve().parents[1]
    if output==root or root in output.parents:
        raise ValueError('review outputs must be outside checkout')
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    def write(name,data):
        (output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    write('protocol.json',protocol())
    write('provenance.json',dict(**provenance(),validation_panel_hash=panel_fingerprint(),
        scientific_training_runs=0,final_test_worlds_instantiated=0))
    policies={'untrained-s0':Policy(seed=0),'untrained-s1':Policy(seed=1),
        'heuristic':HandDesigned(),'fixed-care':CareControl(),
        'abandonment':CareControl('abandonment'),'forage-care':CareControl('forage_care'),
        'uniform':Uniform()}
    timings={}
    for name,policy in policies.items():
        start=time.perf_counter()
        write(name+'.json',evaluate_care(CareTask(),policy))
        timings[name]=time.perf_counter()-start
        print(name,round(timings[name],3),flush=True)
    smoke_records=[]
    if smoke:
        # Exactly one four-world update per algorithm/architecture, then verify
        # restoration (technical validation only; tests check continuation).
        for algorithm in ('reinforce','reinforce_value'):
            for architecture in ('linear','original','deep'):
                exp=CareExperiment(LearnerConfig(algorithm=algorithm,architecture=architecture))
                initial_hash=exp.checkpoint(output/f'{algorithm}-{architecture}-initial.pt')
                start=time.perf_counter();row=exp.batch(validate=False)
                checksum=exp.checkpoint(output/f'{algorithm}-{architecture}-one-update.pt')
                restored=CareExperiment.restore(output/f'{algorithm}-{architecture}-one-update.pt',checksum)
                assert restored.training_decisions==exp.training_decisions
                smoke_records.append(dict(algorithm=algorithm,architecture=architecture,
                    initial_checkpoint_sha256=initial_hash,checkpoint_sha256=checksum,
                    decisions=row['training_decisions'],updates=exp.learner.updates,
                    seconds=time.perf_counter()-start,diagnostics=row['diagnostics']))
    write('timing.json',timings);write('smoke.json',smoke_records)
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}
    write('ARTIFACT_SHA256.json',files)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--smoke',action='store_true',help='six single-batch technical updates, never the pilot')
    args=parser.parse_args();print(review(args.output,args.smoke))
