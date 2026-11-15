"""Which cuts make TSRN really faster? Less depth (whole blocks or GRUs removed) vs less width.

Speed does not depend on the weight values, so the models have random weights and nothing is
trained. Like measure_trajectory_latency.py, all models are timed in one session, in `--rounds`
interleaved rounds, and the median is kept, so slow drift of the machine affects them alike.
Every model is first checked: valid output shape and an exact save/reload round trip.

    python measure_depth_latency.py --out_dir <folder>

Writes depth_latency_<gpu>.csv, the environment and a fit of
    latency = a + b x (GRU steps) + c x (GMACs)
which shows whether the GRU steps that run one after another, or the amount of computation,
set the time.
"""
import argparse
import csv
import os
import re
import statistics
import tempfile

import numpy as np
import torch

import tsrn_depth
import tsrn_experiment as exp
import tsrn_pruning

BLOCKS = ['block2', 'block3', 'block4', 'block5', 'block6']
LABELS = {'gpu_b1': 'GPU b1 ms', 'gpu_b64': 'GPU b64 ms', 'cpu1_b1': 'CPU b1 ms'}


def grus(which, blocks=BLOCKS):
    return [f'{block}.{which}' for block in blocks]


VARIANTS = {  # name: (removed blocks and GRUs, conv channel cut, GRU cut)
    'uncut': ([], 0, 0),
    'width: conv 50%': ([], 0.5, 0),
    'width: conv 87.5%': ([], 0.875, 0),
    'width: conv + GRU 50%': ([], 0.5, 0.5),
    'depth: 4 blocks': (BLOCKS[4:], 0, 0),
    'depth: 3 blocks': (BLOCKS[3:], 0, 0),
    'depth: 2 blocks': (BLOCKS[2:], 0, 0),
    'depth: 1 block': (BLOCKS[1:], 0, 0),
    'depth: 0 blocks': (BLOCKS, 0, 0),
    'depth: no vertical GRUs': (grus('gru1'), 0, 0),
    'depth: no horizontal GRUs': (grus('gru2'), 0, 0),
    'depth: no GRUs at all': (grus('gru1') + grus('gru2'), 0, 0),
    'depth: 3 blocks, no vert. GRUs': (BLOCKS[3:] + grus('gru1', BLOCKS[:3]), 0, 0),
    'both: 3 blocks + conv 50%': (BLOCKS[3:], 0.5, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--gpu_repeats', type=int, default=100)
    parser.add_argument('--cpu_repeats', type=int, default=30)
    return parser.parse_args()


def build(parts, ratio, gru_ratio, device):
    exp.set_seed(0)
    model = tsrn_depth.remove_parts(exp.build_tsrn(), parts).eval()
    if ratio or gru_ratio:
        tsrn_pruning.prune(model, ratio, 'magnitude', round_to=8, gru_ratio=gru_ratio)
    return tsrn_pruning.compact_gru_weights(model.to(device))


def check(model, path, device):
    """Valid output, and the saved model rebuilds to exactly the same model."""
    tsrn_pruning.check_structure(model, device)
    exp.save_pruned(model, path)
    difference = tsrn_pruning.max_output_difference(model, exp.load_pruned(path, device), device)
    if difference > 1e-4:
        raise RuntimeError(f'{path}: the saved model does not reload to the same model ({difference:.1e})')


def fit(features, y):
    """Least-squares fit y = a + sum(b_i x_i); returns R^2 and the coefficients."""
    X = np.column_stack([np.ones(len(y))] + features)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ coef
    centred = y - y.mean()
    return 1 - (residual @ residual) / (centred @ centred), coef


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('1. building and checking the models (random weights):', flush=True)
    models, sizes = {}, {}
    with tempfile.TemporaryDirectory() as folder:
        for index, (name, (parts, ratio, gru_ratio)) in enumerate(VARIANTS.items()):
            model = build(parts, ratio, gru_ratio, device)
            check(model, os.path.join(folder, f'{index:02d}.pth'), device)
            models[name] = model
            sizes[name] = {'params': tsrn_pruning.inference_parameters(model),
                           'macs': tsrn_pruning.inference_macs(model), 'gru_steps': tsrn_depth.gru_steps(model)}
            print(f"   {name:32s} {sizes[name]['params']:>9,d} params {sizes[name]['macs'] / 1e6:>5.0f} MMACs "
                  f"{sizes[name]['gru_steps']:>4d} GRU steps  ok", flush=True)

    print(f'2. timing on {exp.gpu_name()} and 1 CPU thread, {args.rounds} interleaved rounds:', flush=True)
    timings = {name: [] for name in models}
    for round_index in range(1, args.rounds + 1):
        for name, model in models.items():
            timings[name].append(exp.measure_latency(model, gpu_repeats=args.gpu_repeats,
                                                     cpu_repeats=args.cpu_repeats))
        print(f'   round {round_index}/{args.rounds} done', flush=True)

    keys = list(timings['uncut'][0])
    medians = {name: {key: statistics.median(t[key] for t in runs) for key in keys} for name, runs in timings.items()}
    rows = []
    for name, (parts, ratio, gru_ratio) in VARIANTS.items():
        row = {'variant': name, 'removed': ' '.join(parts), 'conv_cut': ratio, 'gru_cut': gru_ratio, **sizes[name]}
        for key in keys:
            row[f'{key}_ms'] = medians[name][key]
            row[f'speedup_{key}'] = medians['uncut'][key] / medians[name][key]
        row['gpu'] = exp.gpu_name()
        rows.append(row)

    print()
    print(f'{"variant":32s}{"params":>9s}{"MMACs":>7s}{"steps":>6s}' + ''.join(f'{LABELS[k]:>17s}' for k in keys))
    for row in rows:
        print(f"{row['variant']:32s}{row['params']:>9,d}{row['macs'] / 1e6:>7.0f}{row['gru_steps']:>6d}"
              + ''.join(f"{row[f'{k}_ms']:>9.2f} (x{row[f'speedup_{k}']:.2f})" for k in keys))
    print('steps = GRU time steps per image (both directions); latency = median ms per batch.')

    print()
    print('3. what sets the time? fit over all variants: latency = a + b x GRU steps + c x GMACs')
    steps = np.array([row['gru_steps'] for row in rows], dtype=float)
    gmacs = np.array([row['macs'] / 1e9 for row in rows])
    fits = {}
    for key in keys:
        y = np.array([row[f'{key}_ms'] for row in rows])
        r2_steps, _ = fit([steps], y)
        r2_macs, _ = fit([gmacs], y)
        r2_both, (a, b, c) = fit([steps, gmacs], y)
        fits[key] = {'r2_steps_only': r2_steps, 'r2_macs_only': r2_macs, 'r2_both': r2_both,
                     'ms_constant': a, 'us_per_gru_step': b * 1000, 'ms_per_gmac': c}
        print(f'   {LABELS[key]:11s} R2: steps only {r2_steps:.2f}, MACs only {r2_macs:.2f}, both {r2_both:.2f}'
              f' | {a:.2f} ms + {b * 1000:.1f} us per GRU step + {c:.2f} ms per GMAC')

    tag = re.sub(r'[^a-z0-9]+', '-', exp.gpu_name().lower()).strip('-')
    path = os.path.join(args.out_dir, f'depth_latency_{tag}.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    exp.write_json(os.path.join(args.out_dir, f'depth_latency_{tag}_fit.json'),
                   {'fits': fits, 'environment': exp.environment_info(), 'args': vars(args)})
    print(f'\nsaved {path} and depth_latency_{tag}_fit.json')


if __name__ == '__main__':
    main()
