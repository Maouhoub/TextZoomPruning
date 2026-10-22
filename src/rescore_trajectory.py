"""Re-score every model of trajectory runs with all three recognizers, image by image.

During a run, validation accuracy is measured with CRNN only and test accuracy is kept as
averages. This script evaluates every saved model again (step 0 = the unpruned checkpoint the run
started from) and keeps what ASTER, MORAN and CRNN read on every validation and test image. That
gives:

- a steadier validation signal for the OCR stop rule: the average accuracy of the three readers;
- paired McNemar tests on the 4,373 test images: does a pruned model read significantly fewer
  words than the unpruned one, per reader?

Resumable: each model's result is saved as rescore/stepNN.json in its run folder, and models
already scored are skipped. A model shared by several runs (the unpruned checkpoint) is scored once.

    python rescore_trajectory.py --run_dirs <run folder> [<run folder> ...]

Writes rescore/stepNN.json, rescore/labels.json and rescore/summary.csv in every run folder.
"""
import argparse
import csv
import math
import os
import statistics
from types import SimpleNamespace

import torch

import tsrn_experiment as exp

SETS = ('val',) + exp.SPLIT_ORDER


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_dirs', nargs='+', required=True, help='trajectory run folders')
    parser.add_argument('--config', default='config/super_resolution.yaml')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='MORAN supports at most 256')
    return parser.parse_args()


def mcnemar(reference, candidate):
    """Exact two-sided McNemar test on paired right/wrong outcomes.

    Returns (lost, gained, p): images only the reference reads right, images only the candidate
    reads right, and the probability of a split at least this uneven if both models were equal.
    """
    lost = sum(r and not k for r, k in zip(reference, candidate))
    gained = sum(k and not r for r, k in zip(reference, candidate))
    n = lost + gained
    if n == 0:
        return lost, gained, 1.0
    tail = sum(math.comb(n, i) for i in range(min(lost, gained) + 1))
    return lost, gained, min(1.0, 2 * tail / 2 ** n)


def load_run_data(run_dir, config, eval_batch_size):
    """Validation split of the run (checked against its val_split.json) and the test splits."""
    run_args = exp.read_json(os.path.join(run_dir, 'config.json'))['args']
    split_args = SimpleNamespace(val_fraction=run_args['val_fraction'], split_seed=run_args.get('split_seed', 0),
                                 batch_size=run_args['batch_size'], eval_batch_size=eval_batch_size)
    data = exp.load_data(config.TRAIN, split_args, os.path.join(run_dir, 'val_split.json'), with_test=True)
    return run_args, data


def score_run(run_dir, config, textbase, recognizers, device, args, scored):
    run_args, data = load_run_data(run_dir, config, args.eval_batch_size)
    loaders = {'val': data.val, **data.test}
    out_dir = os.path.join(run_dir, 'rescore')
    os.makedirs(out_dir, exist_ok=True)
    print(f'== {os.path.basename(os.path.normpath(run_dir))}', flush=True)

    for record in exp.read_json(os.path.join(run_dir, 'trajectory.json')):
        path = os.path.join(out_dir, f"step{record['step']:02d}.json")
        if os.path.exists(path):
            continue
        unpruned = record['checkpoint'] is None
        checkpoint = run_args['resume'] if unpruned else os.path.join(run_dir, record['checkpoint'])
        key = (os.path.abspath(checkpoint), tuple(data.val_indices))
        if key not in scored:
            model = exp.load_tsrn(checkpoint, device) if unpruned else exp.load_pruned(checkpoint, device)
            scored[key] = {name: exp.predict(model, loader, textbase, recognizers, device)
                           for name, loader in loaders.items()}
        result = scored[key]
        labels_path = os.path.join(out_dir, 'labels.json')
        if not os.path.exists(labels_path):
            exp.write_json(labels_path, {name: result[name]['labels'] for name in SETS})
        exp.write_json(path, {'step': record['step'], 'removed_pct': record['removed_pct'], 'checkpoint': checkpoint,
                              'sets': {name: {k: v for k, v in result[name].items() if k != 'labels'}
                                       for name in SETS}})
        print(f"  step {record['step']} scored", flush=True)
    write_summary(out_dir)


def write_summary(out_dir):
    labels = exp.read_json(os.path.join(out_dir, 'labels.json'))
    steps = [exp.read_json(os.path.join(out_dir, name))
             for name in sorted(os.listdir(out_dir)) if name.startswith('step') and name.endswith('.json')]

    def correct(step, name, rec):
        return [p == t for p, t in zip(step['sets'][name]['predictions'][rec], labels[name])]

    def test_correct(step, rec):  # the three test splits pooled = the weighted average of the paper
        return [c for split in exp.SPLIT_ORDER for c in correct(step, split, rec)]

    rows = []
    for step in steps:
        row = {'step': step['step'], 'removed_pct': step['removed_pct'], 'val_psnr': step['sets']['val']['psnr']}
        for rec in exp.RECOGNIZERS:
            row[f'val_{rec}'] = 100.0 * statistics.mean(correct(step, 'val', rec))
        row['val_ocr_average'] = statistics.mean(row[f'val_{rec}'] for rec in exp.RECOGNIZERS)
        for rec in exp.RECOGNIZERS:
            row[f'test_{rec}_average'] = 100.0 * statistics.mean(test_correct(step, rec))
        for rec in exp.RECOGNIZERS:
            lost, gained, p = mcnemar(test_correct(steps[0], rec), test_correct(step, rec))
            row.update({f'mcnemar_{rec}_lost': lost, f'mcnemar_{rec}_gained': gained, f'mcnemar_{rec}_p': p})
        rows.append(row)
        print(f"  step {row['step']} ({row['removed_pct']:.1f}% removed): val ASTER/MORAN/CRNN "
              + '/'.join(f"{row[f'val_{rec}']:.1f}" for rec in exp.RECOGNIZERS)
              + f" average {row['val_ocr_average']:.2f} | test McNemar p vs step 0 "
              + ', '.join(f"{rec.upper()} {row[f'mcnemar_{rec}_p']:.3g}" for rec in exp.RECOGNIZERS))

    path = os.path.join(out_dir, 'summary.csv')
    with open(f'{path}.tmp', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(f'{path}.tmp', path)
    print(f'  saved {path}', flush=True)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = exp.load_config(args.config)
    textbase = exp.make_textbase(config, args.eval_batch_size)
    recognizers = {rec: textbase.load_recognizer(rec) for rec in exp.RECOGNIZERS}
    scored = {}
    for run_dir in args.run_dirs:
        score_run(run_dir, config, textbase, recognizers, device, args, scored)


if __name__ == '__main__':
    main()
