"""Pruning pilot for TSRN: one pruning step on the convolutions, a short fine-tune, then measure.

1. Hold out a validation split of the training set (model selection never looks at the test sets).
2. Depth pruning (optional): remove `--remove_blocks` whole residual blocks, chosen one at a time
   as the block whose removal hurts validation reading (CRNN, or PSNR with `--depth_criterion psnr`)
   least, then `--remove_grus` (gru1 = vertical, gru2 = horizontal GRUs) in every remaining block.
   Then prune `--ratio` of the channels of every prunable convolution group, and `--gru_ratio` of
   every GRU group (GRU input features, and GRU hidden units with the trunk channels tied to them;
   0 = GRUs protected). Units are scored with Taylor importance on the image loss, plus the CTC loss of a frozen
   CRNN reading the output when `--score_rec_weight` > 0 (reading-aware pruning).
3. Check the pruned model still outputs a valid 4 x 32 x 128 image.
4. Fine-tune with the unpruned model as a distillation teacher (or `--teacher`, e.g. the full TSRN
   when `--resume` is a shallow TSRN trained from scratch, or `--teacher_lemma`, LEMMA's released
   model), plus the CRNN reading loss when `--ft_rec_weight` > 0.
5. Measure OCR accuracy (ASTER, MORAN, CRNN), PSNR/SSIM, size and latency, before and after.
   What every reader reads on every test image is kept, so pilots can be compared with McNemar
   tests (compare_pilots.py).

Run from src/ after the config points at the data and recognizers:
    python prune_tsrn_pilot.py --resume <tsrn checkpoint> --out_dir <results folder>
A pilot whose results.json exists is skipped.
"""
import argparse
import os

import torch

import lemma_teacher
import tsrn_depth
import tsrn_experiment as exp
import tsrn_pruning
from loss.image_loss import ImageLoss


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--resume', required=True, help='trained TSRN checkpoint (state_dict_G)')
    teacher = parser.add_mutually_exclusive_group()
    teacher.add_argument('--teacher', default=None,
                         help='distillation teacher checkpoint (default: the --resume model before pruning)')
    teacher.add_argument('--teacher_lemma', default=None,
                         help="LEMMA's released checkpoint as the distillation teacher (run only, never trained)")
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--config', default='config/super_resolution.yaml')
    parser.add_argument('--ratio', type=float, default=0.2, help='fraction of channels removed per convolution group')
    parser.add_argument('--gru_ratio', type=float, default=0.0,
                        help='fraction of units removed per GRU group (0 = GRUs and trunk protected)')
    parser.add_argument('--remove_blocks', type=int, default=0,
                        help='whole residual blocks removed (depth pruning), chosen one at a time on validation')
    parser.add_argument('--depth_criterion', default='crnn', choices=['crnn', 'psnr'],
                        help='validation score kept highest when choosing the block to remove')
    parser.add_argument('--remove_grus', default='none', choices=['none', 'gru1', 'gru2'],
                        help='remove this GRU in every remaining block (gru1 = vertical, gru2 = horizontal)')
    parser.add_argument('--round_to', type=int, default=4, help='channels are removed in multiples of this')
    parser.add_argument('--importance', default='taylor', choices=['taylor', 'magnitude'])
    parser.add_argument('--taylor_batches', type=int, default=2)
    parser.add_argument('--score_rec_weight', type=float, default=0.0,
                        help='weight of the CRNN reading (CTC) loss in the Taylor scores (0 = image loss only)')
    parser.add_argument('--ft_rec_weight', type=float, default=0.0,
                        help='weight of the CRNN reading (CTC) loss during fine-tuning (0 = image loss only)')
    parser.add_argument('--epochs', type=int, default=3, help='maximum fine-tuning epochs')
    parser.add_argument('--patience', type=int, default=0,
                        help='stop fine-tuning after this many epochs without validation gain (0 = never)')
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help='smallest validation gain that counts (dB for psnr, points for ocr)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kd_weight', type=float, default=0.1)
    parser.add_argument('--select_by', default='psnr', choices=['psnr', 'ocr', 'last'],
                        help='validation metric that picks the best fine-tuning epoch (ocr = CRNN accuracy; '
                             'last = keep the last epoch, no selection)')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='MORAN supports at most 256')
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--split_seed', type=int, default=0, help='seed of the train/validation split')
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
    results_path = os.path.join(args.out_dir, 'results.json')
    if os.path.exists(results_path):
        print(f'{results_path} exists: pilot already done')
        print_summary(exp.read_json(results_path)['stages'])
        return
    exp.set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = exp.load_config(args.config)
    textbase = exp.make_textbase(config, args.eval_batch_size)

    # 1. Validation split of the training set.
    data = exp.load_data(config.TRAIN, args, os.path.join(args.out_dir, 'val_split.json'), with_test=not args.skip_test)
    print(f'1. split: {data.n_train} train / {data.n_val} validation images', flush=True)
    for checkpoint in filter(None, (args.resume, args.teacher)):
        exp.check_baseline_split(checkpoint, data.val_indices)
    if args.teacher_lemma:
        print('   note: LEMMA was trained on the whole TextZoom training set, our validation images included; '
              'validation only reports progress here, it selects nothing with --select_by last')
    recognizers = {rec: textbase.load_recognizer(rec) for rec in (exp.RECOGNIZERS if data.test else ('crnn',))}
    val_recognizers = {'crnn': recognizers['crnn']}

    student = exp.load_tsrn(args.resume, device)
    if args.teacher_lemma:
        teacher = lemma_teacher.load_lemma(args.teacher_lemma, device)
    else:
        teacher = exp.frozen_copy(exp.load_tsrn(args.teacher, device) if args.teacher else student)

    def measure(model, keep_predictions=False):
        stage = exp.describe(model, device)
        stage['val'] = exp.evaluate(model, data.val, textbase, val_recognizers, device)
        if data.test:
            stage['test'] = exp.evaluate_test(model, data.test, textbase, recognizers, device, keep_predictions)
        return stage

    stages = {}
    print('   measuring the unpruned model...', flush=True)
    stages['baseline'] = measure(student, keep_predictions=True)

    # 2. Depth pruning (whole blocks, then whole GRUs), then one pruning step on the channels.
    depth_selection = []
    if args.remove_blocks:
        print(f'2a. removing {args.remove_blocks} block(s), each time the one whose removal keeps validation '
              f'{args.depth_criterion} highest:', flush=True)

        def depth_score(model):
            val = exp.evaluate(model, data.val, textbase, val_recognizers, device)
            return val['accuracy']['crnn'] if args.depth_criterion == 'crnn' else val['psnr']

        depth_selection = tsrn_depth.choose_blocks(student, args.remove_blocks, depth_score)
    if args.remove_grus != 'none':
        tsrn_depth.remove_parts(student, [f'{name}.{args.remove_grus}' for name in tsrn_depth.remaining_blocks(student)])
    if depth_selection or args.remove_grus != 'none':
        print(f'2a. removed: {" ".join(tsrn_depth.removed_parts(student))} '
              f'({tsrn_depth.gru_steps(student)} GRU steps left of 800)', flush=True)

    criterion = ImageLoss(gradient=True, loss_weight=[1, 1e-4]).to(device)
    recognition = exp.make_recognition_loss(textbase) if args.score_rec_weight or args.ft_rec_weight else None
    # Reseed after loading the extra CRNN (its construction draws random numbers), so every pilot
    # with the same seed scores channels on the same batches and fine-tunes on the same batch order.
    exp.set_seed(args.seed)
    if args.importance == 'taylor':
        score_loss = exp.scoring_loss(criterion, recognition, args.score_rec_weight)
        tsrn_pruning.accumulate_gradients(student, data.train, score_loss, args.taylor_batches, device)
    report = tsrn_pruning.prune(student, args.ratio, args.importance, args.round_to, gru_ratio=args.gru_ratio)
    print(f'2. pruned {args.ratio:.0%} per convolution group and {args.gru_ratio:.0%} per GRU group '
          f'({args.importance}, reading-loss weight {args.score_rec_weight:g}):')
    for name, (before, after) in report.items():
        print(f'     {name:16s} {before} -> {after}')

    # 3. Structural check.
    print(f'3. output shape after pruning: {tsrn_pruning.check_structure(student, device)}', flush=True)
    stages['pruned, no fine-tune'] = {'params': tsrn_pruning.inference_parameters(student),
                                      'macs': tsrn_pruning.inference_macs(student),
                                      'val': exp.evaluate(student, data.val, textbase, val_recognizers, device)}

    # 4. Fine-tune with the unpruned model as teacher.
    print(f'4. fine-tuning for up to {args.epochs} epoch(s), reading-loss weight {args.ft_rec_weight:g}, '
          f'best epoch by validation {args.select_by}:', flush=True)
    exp.set_seed(args.seed + 1)
    history, best_epoch = exp.fine_tune(student, teacher, data.train, data.val, textbase, val_recognizers, args,
                                        device, recognition=recognition)

    # 5. Measure the pruned model.
    print('5. measuring the pruned model...', flush=True)
    stages['pruned + fine-tune'] = measure(student, keep_predictions=True)

    model_path = os.path.join(args.out_dir, 'tsrn_pruned.pth')
    exp.save_pruned(student, model_path, source=args.resume, ratio=args.ratio, gru_ratio=args.gru_ratio,
                    importance=args.importance, score_rec_weight=args.score_rec_weight,
                    ft_rec_weight=args.ft_rec_weight)
    difference = tsrn_pruning.max_output_difference(student, exp.load_pruned(model_path, device), device)
    print(f'6. reload check: saved model rebuilt, largest output difference {difference:.1e}')
    if difference > 1e-4:
        raise RuntimeError(f'{model_path} does not reload to the same model')
    exp.write_json(results_path, {'args': vars(args), 'environment': exp.environment_info(), 'pruning': report,
                                  'channels': tsrn_pruning.channel_sizes(student, gru=True),
                                  'depth': tsrn_depth.removed_parts(student), 'depth_selection': depth_selection,
                                  'gru_steps': tsrn_depth.gru_steps(student), 'fine_tune': history,
                                  'best_epoch': best_epoch, 'stages': stages})
    print()
    print_summary(stages)
    print(f'\nSaved results.json, val_split.json and tsrn_pruned.pth to {args.out_dir}')


if __name__ == '__main__':
    main()
