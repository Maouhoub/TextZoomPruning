"""Compare pruning pilots: test accuracy per reader, and McNemar tests against a reference pilot.

The first folder is the reference (e.g. image-loss scoring); every other pilot is compared with it
image by image on the 4,373 test images, per reader. Pilots must prune the same baseline to the same size.

    python compare_pilots.py --dirs <reference pilot> <pilot> [<pilot> ...] [--out comparison.csv]
"""
import argparse
import csv
import os

import tsrn_experiment as exp
from rescore_trajectory import mcnemar

PRUNED, BASELINE = 'pruned + fine-tune', 'baseline'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dirs', nargs='+', required=True, help='pilot folders; the first is the reference')
    parser.add_argument('--out', default=None, help='CSV file for the table (default: next to the reference)')
    return parser.parse_args()


def test_correct(stage, rec):
    """Right/wrong on every test image (splits pooled), for one reader."""
    predictions = stage['test']['predictions']
    return [read == label for split in exp.SPLIT_ORDER
            for read, label in zip(predictions[split]['read'][rec], predictions[split]['labels'])]


def main():
    args = parse_args()
    pilots = [(os.path.basename(os.path.normpath(d)), exp.read_json(os.path.join(d, 'results.json')))
              for d in args.dirs]
    reference_name, reference = pilots[0]
    sizes = {name: result['stages'][PRUNED]['params'] for name, result in pilots}
    if len(set(sizes.values())) > 1:
        print(f'WARNING: the pilots have different sizes {sizes}; comparisons are not like for like')

    unpruned = reference['stages'][BASELINE]['test']['average']['accuracy']
    print('unpruned: ' + ', '.join(f'{rec.upper()} {unpruned[rec]:.2f}' for rec in exp.RECOGNIZERS))
    rows = []
    for name, result in pilots:
        stage = result['stages'][PRUNED]
        accuracy = stage['test']['average']['accuracy']
        row = {'pilot': name, 'params': stage['params'], 'best_epoch': result.get('best_epoch'),
               'epochs_run': len(result['fine_tune']), 'val_psnr': stage['val']['psnr'],
               'val_crnn': stage['val']['accuracy']['crnn']}
        for rec in exp.RECOGNIZERS:
            row[f'test_{rec}'] = accuracy[rec]
            row[f'delta_{rec}_vs_unpruned'] = accuracy[rec] - unpruned[rec]
        for rec in exp.RECOGNIZERS:
            lost, gained, p = mcnemar(test_correct(reference['stages'][PRUNED], rec), test_correct(stage, rec))
            row.update({f'{rec}_lost_vs_ref': lost, f'{rec}_gained_vs_ref': gained, f'{rec}_p_vs_ref': p})
        rows.append(row)
        print(f"{name}\n  test ASTER/MORAN/CRNN "
              + ' / '.join(f"{accuracy[rec]:.2f} ({accuracy[rec] - unpruned[rec]:+.2f})" for rec in exp.RECOGNIZERS)
              + f" | val PSNR {row['val_psnr']:.2f}, epochs {row['epochs_run']}"
              + ('' if name == reference_name else
                 '\n  vs reference: ' + ', '.join(f"{rec.upper()} -{row[f'{rec}_lost_vs_ref']}/+{row[f'{rec}_gained_vs_ref']}"
                                                  f" p={row[f'{rec}_p_vs_ref']:.3g}" for rec in exp.RECOGNIZERS)))

    out = args.out or os.path.join(os.path.dirname(os.path.normpath(args.dirs[0])), 'comparison.csv')
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
