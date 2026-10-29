"""RQ1 trajectory: prune TSRN step by step and record PSNR and OCR accuracy after every step.

Each step removes channels from every prunable convolution group (GRUs protected), fine-tunes
briefly with the unpruned model as teacher, then records size, MACs, latency, validation PSNR and
CRNN accuracy, and test accuracy with ASTER, MORAN and CRNN. There is no stop rule: the schedule
runs to the end so the whole trajectory is visible. Stop rules (PSNR or OCR based) can then be
applied to the recorded validation curve afterwards (RQ2).

Resumable: every finished step is saved in the run folder; running the same command again
continues after the last finished step. An interrupted step is simply redone.

    python prune_tsrn_trajectory.py --resume <unpruned TSRN checkpoint> --experiments_dir <Drive folder>
"""
import argparse
import csv
import os
import time

import torch

import tsrn_experiment as exp
import tsrn_pruning
from loss.image_loss import ImageLoss

EXPERIMENT = 'rq1_trajectory'
# Arguments that may change between sessions of the same run without changing its results.
RESUME_FREE_ARGS = {'experiments_dir', 'config', 'eval_batch_size', 'test_every', 'run_name', 'print_run_dir'}

RUN_README = """# {run_name}

RQ1 pruning trajectory of TSRN on TextZoom (convolutions only, GRUs protected).
Step 0 is the unpruned model; every later step removes more channels, then fine-tunes.

- `config.json`: every argument, the software versions, the GPU and the git commit
- `val_split.json`: the training images held out for validation (model selection only)
- `trajectory.csv`: one row per step (size, MACs, validation and test metrics, latency)
- `trajectory.json`: the same with full detail (per split, per fine-tuning epoch)
- `checkpoints/stepNN_<channels>ch.pth`: the model after each step
  (load with `tsrn_experiment.load_pruned`)
- `run.log`: everything the script printed, across sessions
- `latency_same_session.csv`: all models timed back to back (measure_trajectory_latency.py)
- `figures/`: plots made by the notebook

Latency in `trajectory.csv` is measured during the run and may come from different Colab
sessions; use `latency_same_session.csv` for the paper.
"""


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--resume', required=True, help='unpruned TSRN checkpoint, also the distillation teacher')
    parser.add_argument('--experiments_dir', required=True, help='root folder for all experiment results')
    parser.add_argument('--run_name', default=None, help='folder name; by default built from the settings')
    parser.add_argument('--print_run_dir', action='store_true', help='print the run folder and exit')
    parser.add_argument('--config', default='config/super_resolution.yaml')
    parser.add_argument('--target_ratio', type=float, default=0.875, help='fraction of channels removed at the end')
    parser.add_argument('--steps', type=int, default=7)
    parser.add_argument('--schedule', default='linear', choices=['linear', 'cubic'])
    parser.add_argument('--round_to', type=int, default=8, help='channel counts stay multiples of this')
    parser.add_argument('--importance', default='taylor', choices=['taylor', 'magnitude'])
    parser.add_argument('--taylor_batches', type=int, default=2)
    parser.add_argument('--score_rec_weight', type=float, default=0.0,
                        help='weight of the CRNN reading (CTC) loss in the Taylor scores (0 = image loss only)')
    parser.add_argument('--ft_rec_weight', type=float, default=0.0,
                        help='weight of the CRNN reading (CTC) loss during fine-tuning (0 = image loss only)')
    parser.add_argument('--epochs', type=int, default=3, help='maximum fine-tuning epochs after each step')
    parser.add_argument('--patience', type=int, default=0,
                        help='stop fine-tuning after this many epochs without validation gain (0 = never)')
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help='smallest validation gain that counts (dB for psnr, points for ocr)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kd_weight', type=float, default=0.1)
    parser.add_argument('--select_by', default='psnr', choices=['psnr', 'ocr'],
                        help='validation metric that picks the best fine-tuning epoch (ocr = CRNN accuracy)')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='MORAN supports at most 256')
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--split_seed', type=int, default=0,
                        help='seed of the train/validation split; keep it equal to the clean baseline training')
    parser.add_argument('--seed', type=int, default=0, help='seed of the pruning and fine-tuning')
    parser.add_argument('--test_every', type=int, default=1, help='test-set evaluation every N steps (and last)')
    return parser


def default_run_name(args):
    # Runs from the community checkpoint keep their original names; other baselines are named first.
    baseline = exp.baseline_name(args.resume)
    model = 'tsrn' if baseline == 'community' else f'tsrn-{baseline}'
    fine_tuning = (f'ft{args.epochs}ep' + (f'-es{args.patience}' if args.patience else '')
                   + (f'-ctc{args.ft_rec_weight:g}' if args.ft_rec_weight else ''))
    importance = args.importance + (f'-ctc{args.score_rec_weight:g}' if args.score_rec_weight else '')
    if args.taylor_batches != 2:
        importance += f'-tb{args.taylor_batches}'
    return (f'{model}_conv-only_{importance}_{args.schedule}-{args.steps}steps-to{round(args.target_ratio * 100)}pct'
            f'_round{args.round_to}_{fine_tuning}_kd{args.kd_weight:g}_select-{args.select_by}_seed{args.seed}')


def removed_pct(sizes, initial):
    return 100.0 * (1 - sum(sizes.values()) / sum(initial.values()))


def csv_row(record):
    sizes = set(record['channels'].values())
    row = {'step': record['step'], 'removed_pct': round(record['removed_pct'], 2),
           'channels': sizes.pop() if len(sizes) == 1 else 'mixed',
           'params': record['params'], 'macs': record['macs'],
           'val_psnr': record['val']['psnr'], 'val_ssim': record['val']['ssim'],
           'val_crnn': record['val']['accuracy']['crnn']}
    before = record.get('val_before_finetune')
    row['val_psnr_before_finetune'] = before['psnr'] if before else ''
    row['val_crnn_before_finetune'] = before['accuracy']['crnn'] if before else ''
    row['best_epoch'] = record.get('best_epoch') or ''
    row['epochs_run'] = len(record.get('fine_tune') or []) or ''
    test = record.get('test')
    for rec in exp.RECOGNIZERS:
        for split in exp.SPLIT_ORDER + ('average',):
            row[f'test_{rec}_{split}'] = test[split]['accuracy'][rec] if test else ''
    for split in exp.SPLIT_ORDER:
        row[f'test_psnr_{split}'] = test[split]['psnr'] if test else ''
        row[f'test_ssim_{split}'] = test[split]['ssim'] if test else ''
    for key, ms in record['latency_ms'].items():
        row[f'latency_{key}_ms'] = ms
    row['gpu'] = record['gpu']
    row['minutes'] = round(record['minutes'], 1)
    return row


def save_trajectory(run_dir, records):
    exp.write_json(os.path.join(run_dir, 'trajectory.json'), records)
    rows = [csv_row(r) for r in records]
    path = os.path.join(run_dir, 'trajectory.csv')
    with open(f'{path}.tmp', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(f'{path}.tmp', path)


def step_summary(record):
    test = record.get('test')
    test_text = ('test ' + '/'.join(f"{test['average']['accuracy'][r]:.1f}" for r in exp.RECOGNIZERS)
                 + ' (ASTER/MORAN/CRNN)') if test else 'no test'
    latency = record['latency_ms']
    return (f"step {record['step']}: {record['removed_pct']:.1f}% channels removed, {record['params']:,} params | "
            f"val PSNR {record['val']['psnr']:.2f} CRNN {record['val']['accuracy']['crnn']:.2f} | {test_text} | "
            f"GPU b1 {latency.get('gpu_b1', float('nan')):.1f} ms, CPU b1 {latency['cpu1_b1']:.1f} ms")


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_name = args.run_name or default_run_name(args)
    run_dir = os.path.join(args.experiments_dir, EXPERIMENT, run_name)
    if args.print_run_dir:
        print(run_dir)
        return
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'figures'), exist_ok=True)
    log = exp.RunLog(os.path.join(run_dir, 'run.log'))
    log(f"===== session started {time.strftime('%Y-%m-%d %H:%M:%S')} on {exp.gpu_name()}")
    exp.start_or_resume_config(run_dir, args, parser, RESUME_FREE_ARGS, log)
    readme = os.path.join(run_dir, 'README.md')
    if not os.path.exists(readme):
        with open(readme, 'w') as f:
            f.write(RUN_README.format(run_name=run_name))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = exp.load_config(args.config)
    textbase = exp.make_textbase(config, args.eval_batch_size)
    data = exp.load_data(config.TRAIN, args, os.path.join(run_dir, 'val_split.json'), with_test=True)
    log(f'split: {data.n_train} train / {data.n_val} validation images')
    exp.check_baseline_split(args.resume, data.val_indices, log)
    recognizers = {rec: textbase.load_recognizer(rec) for rec in exp.RECOGNIZERS}
    val_recognizers = {'crnn': recognizers['crnn']}

    teacher = exp.frozen_copy(exp.load_tsrn(args.resume, device))
    initial = tsrn_pruning.channel_sizes(teacher)
    schedule = {name: tsrn_pruning.keep_schedule(units, args.target_ratio, args.steps, args.schedule, args.round_to)
                for name, units in initial.items()}
    log(f'channels kept per group after each step: {sorted(set(map(tuple, schedule.values())))}')
    criterion = ImageLoss(gradient=True, loss_weight=[1, 1e-4]).to(device)
    recognition = exp.make_recognition_loss(textbase) if args.score_rec_weight or args.ft_rec_weight else None
    score_loss = exp.scoring_loss(criterion, recognition, args.score_rec_weight)

    def measure(model, step):
        record = exp.describe(model, device)
        record['val'] = exp.evaluate(model, data.val, textbase, val_recognizers, device)
        run_test = step == 0 or step == args.steps or step % args.test_every == 0
        record['test'] = exp.evaluate_test(model, data.test, textbase, recognizers, device) if run_test else None
        return record

    trajectory_path = os.path.join(run_dir, 'trajectory.json')
    records = exp.read_json(trajectory_path) if os.path.exists(trajectory_path) else []
    if records and records[0]['gpu'] != exp.gpu_name():
        log(f"WARNING: step 0 was timed on {records[0]['gpu']}, this session has {exp.gpu_name()}; "
            'latency is not comparable across GPUs, use measure_trajectory_latency.py at the end')

    if not records:
        log('step 0: measuring the unpruned model')
        started = time.time()
        student = exp.load_tsrn(args.resume, device)
        record = measure(student, 0)
        record.update(step=0, removed_pct=0.0, channels=initial, checkpoint=None,
                      minutes=(time.time() - started) / 60)
        records.append(record)
        save_trajectory(run_dir, records)
        log(step_summary(record))
    else:
        last = records[-1]
        student = (exp.load_tsrn(args.resume, device) if last['checkpoint'] is None
                   else exp.load_pruned(os.path.join(run_dir, last['checkpoint']), device))
        log(f"continuing after step {last['step']}")

    for step in range(len(records), args.steps + 1):
        exp.set_seed(args.seed * 1000 + step)  # each step is reproducible, whether or not the run was resumed
        started = time.time()
        targets = {name: kept[step - 1] for name, kept in schedule.items()}
        log(f'step {step}/{args.steps}: pruning to {sorted(set(targets.values()))} channels per group')

        if args.importance == 'taylor':
            tsrn_pruning.accumulate_gradients(student, data.train, score_loss, args.taylor_batches, device)
        tsrn_pruning.prune_to(student, targets, args.importance)
        tsrn_pruning.check_structure(student, device)
        before = exp.evaluate(student, data.val, textbase, val_recognizers, device)
        log(f"  before fine-tuning: val CRNN {before['accuracy']['crnn']:.2f}% PSNR {before['psnr']:.2f}")
        history, best_epoch = exp.fine_tune(student, teacher, data.train, data.val, textbase, val_recognizers,
                                            args, device, log=log, recognition=recognition)

        record = measure(student, step)
        sizes = tsrn_pruning.channel_sizes(student)
        checkpoint = os.path.join('checkpoints', f'step{step:02d}_{min(sizes.values())}ch.pth')
        exp.save_pruned(student, os.path.join(run_dir, checkpoint), step=step, source=args.resume)
        record.update(step=step, removed_pct=removed_pct(sizes, initial), channels=sizes,
                      val_before_finetune=before, fine_tune=history, best_epoch=best_epoch,
                      checkpoint=checkpoint, minutes=(time.time() - started) / 60)
        records.append(record)
        save_trajectory(run_dir, records)  # the step counts as finished only once this is written
        log(step_summary(record))

    log(f'trajectory complete: {run_dir}/trajectory.csv')


if __name__ == '__main__':
    main()
