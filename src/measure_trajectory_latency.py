"""Time every model of a trajectory run back to back, in one session, on one GPU and CPU.

Latency recorded during a run can come from several Colab sessions (and different GPU types)
when the run was resumed. For the paper, all models are timed here in the same session. The
models are timed in `--rounds` interleaved rounds and the median per model is kept, so slow
drift of the machine affects every model alike.

    python measure_trajectory_latency.py --run_dir <run folder>

Step 0 is the unpruned checkpoint the run was started from (config.json), unless `--resume` says
otherwise. Writes latency_same_session.csv in the run folder.
"""
import argparse
import csv
import os
import statistics

import torch

import tsrn_experiment as exp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--resume', default=None,
                        help='unpruned TSRN checkpoint (step 0); default: the one in the run config.json')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--gpu_repeats', type=int, default=100)
    parser.add_argument('--cpu_repeats', type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records = exp.read_json(os.path.join(args.run_dir, 'trajectory.json'))
    unpruned = args.resume or exp.read_json(os.path.join(args.run_dir, 'config.json'))['args']['resume']
    print(f'step 0: {unpruned}')
    models = {r['step']: (exp.load_tsrn(unpruned, device) if r['checkpoint'] is None
                          else exp.load_pruned(os.path.join(args.run_dir, r['checkpoint']), device))
              for r in records}

    timings = {step: [] for step in models}
    for round_index in range(1, args.rounds + 1):
        for step, model in models.items():
            timings[step].append(exp.measure_latency(model, gpu_repeats=args.gpu_repeats, cpu_repeats=args.cpu_repeats))
        print(f'round {round_index}/{args.rounds} done', flush=True)

    rows = []
    keys = list(timings[0][0])
    baseline = {key: statistics.median(t[key] for t in timings[0]) for key in keys}
    for record in records:
        row = {'step': record['step'], 'removed_pct': round(record['removed_pct'], 2),
               'params': record['params'], 'macs': record['macs'],
               'macs_ratio': record['macs'] / records[0]['macs']}
        for key in keys:
            median = statistics.median(t[key] for t in timings[record['step']])
            row[f'latency_{key}_ms'] = median
            row[f'speedup_{key}'] = baseline[key] / median
        row['gpu'] = exp.gpu_name()
        rows.append(row)
        print(f"step {row['step']}: MACs x{row['macs_ratio']:.2f} | "
              + ' | '.join(f"{key} {row[f'latency_{key}_ms']:.2f} ms (speed-up x{row[f'speedup_{key}']:.2f})"
                           for key in keys))

    path = os.path.join(args.run_dir, 'latency_same_session.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    exp.write_json(os.path.join(args.run_dir, 'latency_same_session_environment.json'), exp.environment_info())
    print(f'saved {path}')


if __name__ == '__main__':
    main()
