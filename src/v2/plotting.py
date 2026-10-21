"""Learning curves from intermediate validation; no smoothing or pooled inference."""
import json
from pathlib import Path


def learning_curves(log_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = [json.loads(line) for line in Path(log_path).read_text().splitlines() if line]
    if not records:
        raise ValueError('empty training log')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot([r['training_decisions'] for r in records],
                 [r['episode_metrics']['total_reward'] for r in records], marker='.')
    axes[0].set_ylabel('Training colony reward (undiscounted)')
    validation = [r for r in records if r['validation'] is not None]
    for ax, key, label in zip(axes[1:], ('independence_fraction', 'attendance_fraction'),
                             ('Validation eligible pup independence', 'Validation attendance / care exposure')):
        ax.plot([r['training_decisions'] for r in validation],
                [float('nan') if r['validation']['summary'][key] is None else
                 r['validation']['summary'][key] for r in validation], marker='.')
        ax.set_ylabel(label)
        ax.set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel('Training adult decisions (validation excluded)')
    fig.tight_layout()
    # Protect existing plots as well as original results.
    with Path(output).open('xb') as stream:
        fig.savefig(stream, format=Path(output).suffix.lstrip('.') or 'png')
    plt.close(fig)
    return Path(output)
