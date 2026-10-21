"""Phase 1 bounded smoke command. Full campaigns and PPO require separate review."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from v2.experiment import run_smoke
from v2.plotting import learning_curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='new directory outside preserved results/')
    parser.add_argument('--architecture', choices=('linear', 'original', 'deep'), default='original')
    parser.add_argument('--algorithm', choices=('reinforce', 'reinforce_value'), default='reinforce')
    parser.add_argument('--reward', choices=('survival', 'offspring'), default='offspring')
    args = parser.parse_args()
    result = run_smoke(args.out, args.architecture, args.algorithm, args.reward)
    learning_curves(Path(args.out)/'training.jsonl', Path(args.out)/'learning-smoke.png')
    print(f'Technical smoke only: {result.training_decisions} training decisions, '
          f'{result.learner.updates} updates; final test block untouched')


if __name__ == '__main__':
    main()
