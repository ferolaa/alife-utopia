"""Run only the frozen eight-condition, original-MLP v2 learning pilot."""
import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from v2.pilot import run_pilot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,help='new output directory outside checkout')
    args=parser.parse_args()
    torch.set_num_threads(2)
    result=run_pilot(args.out)
    print(f'Completed {len(result["runs"])} approved runs; no final-test worlds used.')


if __name__=='__main__':
    main()
