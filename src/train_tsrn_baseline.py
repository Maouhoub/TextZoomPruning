"""Clean TSRN baseline: the original TextZoom training recipe, trained only on the 95% training split.

The community checkpoint was trained on the whole TextZoom training set, so it has seen our
validation images, and the original code keeps the checkpoint that scores best on the test sets.
This script keeps the original recipe (main.py --arch tsrn --STN --mask --gradient with
config/super_resolution.yaml: MSE + 1e-4 gradient prior loss scaled by 100, Adam with lr 1e-3 and
beta1 0.5, gradient clipping 0.25, batch 512, 500 epochs) and changes only what keeps it clean:

- it trains on the train part of the split shared with the pruning runs (val_split.json);
- nothing is selected on the test sets: the final weights are the baseline. The validation curve
  (PSNR, ASTER/MORAN/CRNN accuracy) is recorded every `--val_every` epochs, and the best
  validation checkpoints are kept only to check that the final weights did not degrade;
- the test sets are evaluated once, at the end, on the final weights.

`--micro_batch` splits every batch into smaller pieces whose gradients add up before the optimizer
step, so a 16 GB GPU (T4) can follow the batch-512 recipe; only the BatchNorm statistics differ
slightly, because they are computed per piece.

Resumable: every `--val_every` epochs the full training state is saved in the run folder; running
the same command again continues from there, also on another GPU type.

    python train_tsrn_baseline.py --experiments_dir <Drive folder> [--micro_batch 256]
"""
import argparse
import csv
import os
import statistics
import time

import torch
import torch.nn as nn
import torch.optim as optim

import tsrn_experiment as exp
import tsrn_pruning
from loss.image_loss import ImageLoss

EXPERIMENT = 'baseline'
# Arguments that may change between sessions of the same run without changing its results.
RESUME_FREE_ARGS = {'experiments_dir', 'config', 'eval_batch_size', 'run_name', 'print_run_dir', 'workers',
                    'micro_batch'}
PAPER_TSRN = {'aster': 58.3, 'moran': 54.8, 'crnn': 41.4}  # arXiv 2005.03341, Table 4, weighted average
COMMUNITY_TSRN = {'aster': 57.4, 'moran': 53.6, 'crnn': 43.1}  # our evaluation of the community checkpoint

RUN_README = """# {run_name}

Clean TSRN baseline: the original TextZoom training recipe, trained only on the train part of
`val_split.json` (the validation images are never seen), no model selection on the test sets.

- `config.json`: every argument, the software versions, the GPU and the git commit
- `val_split.json`: the training images held out for validation (same split as the pruning runs)
- `checkpoints/final.pth`: **the baseline** (weights after the last epoch); use it as `--resume`
  of prune_tsrn_trajectory.py
- `checkpoints/best_val_ocr.pth`, `best_val_psnr.pth`: best validation epochs, only to check that
  the final weights did not degrade
- `history.csv` / `history.json`: training loss every epoch, validation metrics every few epochs
- `test_final.json`: the only test-set evaluation, on the final weights
- `state/last.pth`: the training state used to resume (can be deleted once the run is finished)
- `run.log`: everything the script printed, across sessions
- `figures/`: plots made by the notebook
"""


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiments_dir', required=True, help='root folder for all experiment results')
    parser.add_argument('--run_name', default=None, help='folder name; by default built from the settings')
    parser.add_argument('--print_run_dir', action='store_true', help='print the run folder and exit')
    parser.add_argument('--config', default='config/super_resolution.yaml')
    # Defaults below are the values of config/super_resolution.yaml in the original repository.
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512, help='images per optimizer step')
    parser.add_argument('--micro_batch', type=int, default=0,
                        help='split every batch into pieces of this size to save GPU memory (0 = no split)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--val_every', type=int, default=5, help='validate and save the training state every N epochs')
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--split_seed', type=int, default=0,
                        help='seed of the train/validation split (the pruning runs use the same)')
    parser.add_argument('--seed', type=int, default=0, help='seed of the weight initialisation and the batch order')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='MORAN supports at most 256')
    parser.add_argument('--workers', type=int, default=None, help='data loading processes (default: config value)')
    return parser


def default_run_name(args):
    return (f'tsrn_clean-baseline_train{round((1 - args.val_fraction) * 100)}pct-split{args.split_seed}'
            f'_bs{args.batch_size}_lr{args.lr:g}_beta{args.beta1:g}_{args.epochs}ep_seed{args.seed}')


def save_history(run_dir, history):
    exp.write_json(os.path.join(run_dir, 'history.json'), history)
    rows = []
    for record in history:
        val = record.get('val')
        row = {'epoch': record['epoch'], 'loss': record['loss'], 'seconds': round(record['seconds'], 1),
               'val_psnr': val['psnr'] if val else '', 'val_ssim': val['ssim'] if val else ''}
        for rec in exp.RECOGNIZERS:
            row[f'val_{rec}'] = val['accuracy'][rec] if val else ''
        row['val_ocr_average'] = val['ocr_average'] if val else ''
        rows.append(row)
    path = os.path.join(run_dir, 'history.csv')
    with open(f'{path}.tmp', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(f'{path}.tmp', path)


def test_summary(result):
    average = result['test']['average']['accuracy']
    lines = ['test (final weights), weighted average accuracy %: '
             + ', '.join(f'{rec.upper()} {average[rec]:.2f} (paper {PAPER_TSRN[rec]}, community {COMMUNITY_TSRN[rec]})'
                         for rec in exp.RECOGNIZERS),
             'test PSNR: ' + ', '.join(f"{split} {result['test'][split]['psnr']:.2f}" for split in exp.SPLIT_ORDER),
             f"validation OCR average: final {result['final_val']['ocr_average']:.2f}, best "
             f"{result['best_val']['ocr']['score']:.2f} at epoch {result['best_val']['ocr']['epoch']}"]
    return '\n'.join(lines)


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_name = args.run_name or default_run_name(args)
    run_dir = os.path.join(args.experiments_dir, EXPERIMENT, run_name)
    if args.print_run_dir:
        print(run_dir)
        return
    for folder in ('checkpoints', 'state', 'figures'):
        os.makedirs(os.path.join(run_dir, folder), exist_ok=True)
    log = exp.RunLog(os.path.join(run_dir, 'run.log'))
    log(f"===== session started {time.strftime('%Y-%m-%d %H:%M:%S')} on {exp.gpu_name()}")
    exp.start_or_resume_config(run_dir, args, parser, RESUME_FREE_ARGS, log)
    readme = os.path.join(run_dir, 'README.md')
    if not os.path.exists(readme):
        with open(readme, 'w') as f:
            f.write(RUN_README.format(run_name=run_name))

    test_path = os.path.join(run_dir, 'test_final.json')
    if os.path.exists(test_path):
        log('run already finished')
        log(test_summary(exp.read_json(test_path)))
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = exp.load_config(args.config)
    if args.workers is not None:
        config.TRAIN.workers = args.workers
    textbase = exp.make_textbase(config, args.eval_batch_size)
    split_path = os.path.join(run_dir, 'val_split.json')
    data = exp.load_data(config.TRAIN, args, split_path, with_test=True)
    log(f'split: {data.n_train} train / {data.n_val} validation images')
    split = exp.read_json(split_path)
    recognizers = {rec: textbase.load_recognizer(rec) for rec in exp.RECOGNIZERS}
    # Own generator, reseeded every epoch: the batch order depends only on the seed and the epoch,
    # so a resumed run sees the same batches as an uninterrupted one.
    generator = torch.Generator()
    train_loader = exp.make_loader(data.train.dataset, args.batch_size, config.TRAIN, shuffle=True,
                                   generator=generator, persistent=True)

    exp.set_seed(args.seed)
    model = exp.build_tsrn().to(device)
    criterion = ImageLoss(gradient=True, loss_weight=[1, 1e-4]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    state_path = os.path.join(run_dir, 'state', 'last.pth')
    history, best, first_epoch = [], {'ocr': None, 'psnr': None}, 1
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history, best, first_epoch = state['history'], state['best'], state['epoch'] + 1
        log(f"continuing after epoch {state['epoch']}")
    tsrn_pruning.compact_gru_weights(model)
    name = 'clean' if args.seed == 0 else f'clean-seed{args.seed}'

    def save_weights(path, epoch, label):
        exp.save_atomic({'state_dict_G': model.state_dict(),
                         'info': {'name': label, 'run_name': run_name, 'epoch': epoch, 'val_split': split}}, path)

    def train_one_epoch(epoch):
        model.train()
        generator.manual_seed(args.seed * 100_000 + epoch)
        total, batches = torch.zeros((), device=device), 0
        for images_hr, images_lr, _ in train_loader:
            images_lr, images_hr = images_lr.to(device), images_hr.to(device)
            optimizer.zero_grad()
            piece = args.micro_batch or len(images_lr)
            for lr_part, hr_part in zip(images_lr.split(piece), images_hr.split(piece)):
                # Loss of interfaces/super_resolution.py; each piece is weighted by its share of the
                # batch, so the summed gradients equal the full-batch gradient.
                loss = criterion(model(lr_part), hr_part) * 100 * len(lr_part) / len(images_lr)
                loss.backward()
                total += loss.detach()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            batches += 1
        return (total / batches).item()

    session_start = time.time()
    for epoch in range(first_epoch, args.epochs + 1):
        started = time.time()
        record = {'epoch': epoch, 'loss': train_one_epoch(epoch), 'seconds': time.time() - started}
        message = f"epoch {epoch}/{args.epochs}: loss {record['loss']:.4f} ({record['seconds']:.0f} s)"
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val = exp.evaluate(model, data.val, textbase, recognizers, device)
            val['ocr_average'] = statistics.mean(val['accuracy'].values())
            record['val'] = val
            for key, score in (('ocr', val['ocr_average']), ('psnr', val['psnr'])):
                if best[key] is None or score > best[key]['score']:
                    best[key] = {'epoch': epoch, 'score': score}
                    save_weights(os.path.join(run_dir, 'checkpoints', f'best_val_{key}.pth'), epoch,
                                 f'{name}-best-val-{key}')
            hours_left = (args.epochs - epoch) * (time.time() - session_start) / (epoch - first_epoch + 1) / 3600
            message += (f" | val PSNR {val['psnr']:.2f}, "
                        + '/'.join(f"{val['accuracy'][rec]:.1f}" for rec in exp.RECOGNIZERS)
                        + f" (ASTER/MORAN/CRNN), average {val['ocr_average']:.2f} | about {hours_left:.1f} h left")
        history.append(record)
        log(message)
        if 'val' in record:
            save_history(run_dir, history)
            exp.save_atomic({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                             'history': history, 'best': best}, state_path)

    final_path = os.path.join(run_dir, 'checkpoints', 'final.pth')
    save_weights(final_path, args.epochs, name)
    log(f'saved the baseline: {final_path}')
    log('testing the final weights (the only evaluation on the test sets)')
    result = {'checkpoint': 'checkpoints/final.pth', 'epoch': args.epochs,
              'test': exp.evaluate_test(model, data.test, textbase, recognizers, device),
              'final_val': history[-1]['val'], 'best_val': best}
    exp.write_json(test_path, result)
    log(test_summary(result))


if __name__ == '__main__':
    main()
