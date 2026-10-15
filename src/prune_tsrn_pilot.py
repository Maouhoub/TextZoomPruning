"""Pruning pilot for TSRN: one pruning step on the convolutions, a short fine-tune, then measure.

1. Hold out a validation split of the training set (model selection never looks at the test sets).
2. Prune `--ratio` of the channels of every prunable convolution group (GRUs are protected).
3. Check the pruned model still outputs a valid 4 x 32 x 128 image.
4. Fine-tune for a few epochs with the unpruned model as a distillation teacher.
5. Measure OCR accuracy (ASTER, MORAN, CRNN), PSNR/SSIM, size and latency, before and after.

Run from src/ after the config points at the data and recognizers:
    python prune_tsrn_pilot.py --resume <tsrn checkpoint> --out_dir <results folder>
"""
import argparse
import os

import torch

import tsrn_experiment as exp
import tsrn_pruning
from loss.image_loss import ImageLoss


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--resume', required=True, help='trained TSRN checkpoint (state_dict_G)')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--config', default='config/super_resolution.yaml')
    parser.add_argument('--ratio', type=float, default=0.2, help='fraction of channels removed per group')
    parser.add_argument('--round_to', type=int, default=4, help='channels are removed in multiples of this')
    parser.add_argument('--importance', default='taylor', choices=['taylor', 'magnitude'])
    parser.add_argument('--taylor_batches', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kd_weight', type=float, default=0.1)
    parser.add_argument('--select_by', default='psnr', choices=['psnr', 'ocr'],
                        help='validation metric that picks the best fine-tuning epoch (ocr = CRNN accuracy)')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='MORAN supports at most 256')
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip_test', action='store_true', help='only evaluate on the validation split')
    return parser.parse_args()


def print_summary(stages):
    header = (f"{'':24s}{'params':>9s}{'MMACs':>8s}{'val CRNN':>9s}{'val PSNR':>9s}"
              f"{'ASTER':>7s}{'MORAN':>7s}{'CRNN':>7s}{'GPU b1':>8s}{'GPU b64':>8s}{'CPU b1':>8s}")
    print(header)

    def cell(value, fmt, width):
        return f'{value:{width}{fmt}}' if value is not None else ' ' * (width - 1) + '-'

    for label, stage in stages.items():
        test = stage.get('test', {}).get('average', {}).get('accuracy', {})
        latency = stage.get('latency_ms', {})
        print(f'{label:24s}'
              + cell(stage.get('params'), ',d', 9) + cell(stage['macs'] / 1e6 if 'macs' in stage else None, '.0f', 8)
              + cell(stage['val']['accuracy']['crnn'], '.2f', 9) + cell(stage['val']['psnr'], '.2f', 9)
              + ''.join(cell(test.get(rec), '.1f', 7) for rec in exp.RECOGNIZERS)
              + ''.join(cell(latency.get(key), '.1f', 8) for key in ('gpu_b1', 'gpu_b64', 'cpu1_b1')))
    print('Accuracy in %, test = weighted average over easy/medium/hard, latency in ms per batch.')


def main():
    args = parse_args()
    exp.set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = exp.load_config(args.config)
    textbase = exp.make_textbase(config, args.eval_batch_size)

    # 1. Validation split of the training set.
    data = exp.load_data(config.TRAIN, args, os.path.join(args.out_dir, 'val_split.json'), with_test=not args.skip_test)
    print(f'1. split: {data.n_train} train / {data.n_val} validation images', flush=True)
    recognizers = {rec: textbase.load_recognizer(rec) for rec in (exp.RECOGNIZERS if data.test else ('crnn',))}
    val_recognizers = {'crnn': recognizers['crnn']}

    student = exp.load_tsrn(args.resume, device)
    teacher = exp.frozen_copy(student)

    def measure(model):
        stage = exp.describe(model, device)
        stage['val'] = exp.evaluate(model, data.val, textbase, val_recognizers, device)
        if data.test:
            stage['test'] = exp.evaluate_test(model, data.test, textbase, recognizers, device)
        return stage

    stages = {}
    print('   measuring the unpruned model...', flush=True)
    stages['baseline'] = measure(student)

    # 2. One pruning step on the convolutions.
    criterion = ImageLoss(gradient=True, loss_weight=[1, 1e-4]).to(device)
    if args.importance == 'taylor':
        tsrn_pruning.accumulate_gradients(student, data.train, criterion, args.taylor_batches, device)
    report = tsrn_pruning.prune(student, args.ratio, args.importance, args.round_to)
    print(f'2. pruned {args.ratio:.0%} per group ({args.importance}):')
    for name, (before, after) in report.items():
        print(f'     {name:16s} {before} -> {after}')

    # 3. Structural check.
    print(f'3. output shape after pruning: {tsrn_pruning.check_structure(student, device)}', flush=True)
    stages['pruned, no fine-tune'] = {'params': tsrn_pruning.inference_parameters(student),
                                      'macs': tsrn_pruning.inference_macs(student),
                                      'val': exp.evaluate(student, data.val, textbase, val_recognizers, device)}

    # 4. Short fine-tune with the unpruned model as teacher.
    print(f'4. fine-tuning for {args.epochs} epoch(s), best epoch by validation {args.select_by}:', flush=True)
    history, _ = exp.fine_tune(student, teacher, data.train, data.val, textbase, val_recognizers, args, device)

    # 5. Measure the pruned model.
    print('5. measuring the pruned model...', flush=True)
    stages['pruned + fine-tune'] = measure(student)

    exp.save_pruned(student, os.path.join(args.out_dir, 'tsrn_pruned.pth'),
                    source=args.resume, ratio=args.ratio, importance=args.importance)
    exp.write_json(os.path.join(args.out_dir, 'results.json'),
                   {'args': vars(args), 'environment': exp.environment_info(), 'pruning': report,
                    'fine_tune': history, 'stages': stages})
    print()
    print_summary(stages)
    print(f'\nSaved results.json, val_split.json and tsrn_pruned.pth to {args.out_dir}')


if __name__ == '__main__':
    main()
