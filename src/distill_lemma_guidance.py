"""Kill test (notebook section 21): replace LEMMA's text-guidance branch with a small network.

LEMMA's text-guidance branch (36.5M of its 37.9M parameters, 54-74% of its time) only produces a
64 x 16 x 64 guidance map for its SR branch. Here GuidanceNet (lemma_fast.py), a small dilated CNN,
makes that map from the same first-layer features, and LEMMA's SR branch keeps its released
weights, frozen.

1. Train only GuidanceNet, on the TextZoom training images, to reproduce LEMMA's guidance map and
   LEMMA's output. This is distillation only: no reader and no HR image is used, so the three test
   readers are all unseen.
2. Test accuracy with ASTER, MORAN and CRNN, and McNemar tests against LEMMA, image by image.
3. Size, and speed in one session: the student, LEMMA, and the TSRN checkpoints given with --compare.
4. Decision rule, fixed before the run. PASS if both hold:
   - the student loses at most 2 points against LEMMA on every reader (our LEMMA: 67.35 / 65.04 /
     56.48), which still reads above every other model in SGENet's Table 1 (TATT: 63.6 / 59.5 / 52.6).
     Merely matching SGENet (61.4 / 57.3 / 49.0) is not enough: SGENet is cheaper than the student,
     so it would be the better deal;
   - it is at least x2 faster than LEMMA on CPU (1 thread, 1 image) and on GPU batch 64.
   GPU batch 1 is reported.

    python distill_lemma_guidance.py --lemma <LEMMA-release.pth> --width 64 --out_dir <folder>

A run whose results.json exists is skipped (an interrupted run starts again).
"""
import argparse
import os
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import compare_pilots
import evaluate_lemma
import lemma_fast
import lemma_teacher
import tsrn_experiment as exp
import tsrn_pruning
from rescore_trajectory import mcnemar

SGENET = {'aster': 61.4, 'moran': 57.3, 'crnn': 49.0}  # SGENet (ICASSP 2024) Table 1, same three readers
MAX_LOSS_VS_LEMMA = 2.0  # points per reader
SPEED_SETTINGS = ('cpu1_b1', 'gpu_b64')  # where LEMMA's guidance branch takes 72% and 74% of the time
MIN_SPEEDUP = 2.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lemma', required=True, help="LEMMA's released checkpoint")
    parser.add_argument('--lemma_results', default=None,
                        help="lemma_results.json from evaluate_lemma.py, to reuse LEMMA's test predictions")
    parser.add_argument('--width', type=int, default=64, help='channels of GuidanceNet')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--config', default='config/super_resolution.yaml')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--map_weight', type=float, default=1.0,
                        help='weight of the guidance-map loss (MSE as a fraction of the map variance)')
    parser.add_argument('--out_weight', type=float, default=100.0,
                        help="weight of the output loss (MSE to LEMMA's SR image)")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=256, help='MORAN supports at most 256')
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--split_seed', type=int, default=0, help='seed of the train/validation split')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--compare', nargs='*', default=[], help='TSRN checkpoints to time in the same session')
    parser.add_argument('--compare_names', nargs='*', default=None, help='labels for --compare')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--gpu_repeats', type=int, default=100)
    parser.add_argument('--cpu_repeats', type=int, default=20)
    args = parser.parse_args()
    if args.compare_names is not None and len(args.compare_names) != len(args.compare):
        parser.error('give one --compare_names label per --compare checkpoint')
    return args


def distillation_terms(student, teacher, x):
    """Guidance-map error (MSE as a fraction of the map's variance: 0 = exact copy, about 1 = no better
    than its mean) and output MSE to LEMMA's SR image. The frozen first layer runs without gradients."""
    with torch.no_grad():
        target_sr, target_map = lemma_teacher.outputs_with_guidance(teacher, x)
        features = student.block1(x)
    predicted_map = student.guidance(features)
    map_error = F.mse_loss(predicted_map, target_map) / target_map.var()
    output_mse = F.mse_loss(lemma_teacher.sr_branch(student, features, predicted_map), target_sr)
    return map_error, output_mse


@torch.no_grad()
def validate(student, teacher, loader, device):
    student.eval()
    terms = [distillation_terms(student, teacher, images_lr.to(device)) for _, images_lr, _ in loader]
    return {'map_error': statistics.mean(t[0].item() for t in terms),
            'output_mse': statistics.mean(t[1].item() for t in terms)}


def train(student, teacher, data, args, device, textbase, crnn):
    """Train GuidanceNet only, cosine learning rate; the last epoch is kept (no selection)."""
    optimizer = optim.Adam(student.guidance.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(data.train))
    history = []
    for epoch in range(1, args.epochs + 1):
        lemma_fast.train_guidance_only(student)
        map_errors, output_mses = [], []
        for _, images_lr, _ in data.train:
            map_error, output_mse = distillation_terms(student, teacher, images_lr.to(device))
            loss = args.map_weight * map_error + args.out_weight * output_mse
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.guidance.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            map_errors.append(map_error.item())
            output_mses.append(output_mse.item())
        val = validate(student, teacher, data.val, device)
        val['crnn'] = exp.evaluate(student, data.val, textbase, {'crnn': crnn}, device)['accuracy']['crnn']
        history.append({'epoch': epoch, 'train_map_error': statistics.mean(map_errors),
                        'train_output_mse': statistics.mean(output_mses), 'val': val})
        print(f"  epoch {epoch}: train map error {history[-1]['train_map_error']:.3f}, output MSE "
              f"{history[-1]['train_output_mse']:.5f} | val map error {val['map_error']:.3f}, output MSE "
              f"{val['output_mse']:.5f}, CRNN {val['crnn']:.2f}%", flush=True)
    student.eval()
    return history


def check_same_images(test, lemma_test):
    for split in exp.SPLIT_ORDER:
        if test['predictions'][split]['labels'] != lemma_test['predictions'][split]['labels']:
            raise RuntimeError(f'the LEMMA results are on other {split} test images')


def compare_with_lemma(test, lemma_test):
    """Accuracy change vs LEMMA and McNemar test on the 4,373 test images, per reader."""
    check_same_images(test, lemma_test)
    result = {}
    for rec in exp.RECOGNIZERS:
        lost, gained, p = mcnemar(compare_pilots.test_correct({'test': lemma_test}, rec),
                                  compare_pilots.test_correct({'test': test}, rec))
        result[rec] = {'delta': test['average']['accuracy'][rec] - lemma_test['average']['accuracy'][rec],
                       'lost': lost, 'gained': gained, 'p': p}
    return result


def decide(accuracy, lemma_accuracy, speedup):
    checked = [key for key in SPEED_SETTINGS if key in speedup]  # no GPU: CPU only
    near_lemma = all(accuracy[rec] >= lemma_accuracy[rec] - MAX_LOSS_VS_LEMMA for rec in exp.RECOGNIZERS)
    faster = bool(checked) and all(speedup[key] >= MIN_SPEEDUP for key in checked)
    return {'accuracy': accuracy, 'lemma_accuracy': lemma_accuracy, 'sgenet': SGENET,
            'within_2_points_of_lemma': near_lemma,
            'reads_at_least_sgenet': all(accuracy[rec] >= SGENET[rec] for rec in exp.RECOGNIZERS),
            'speedup_vs_lemma': speedup, 'speed_settings_checked': checked,
            'at_least_x2_faster': faster, 'verdict': 'PASS' if near_lemma and faster else 'FAIL'}


def print_decision(decision):
    accuracy, lemma, speedup = decision['accuracy'], decision['lemma_accuracy'], decision['speedup_vs_lemma']
    print(f'decision rule: at most {MAX_LOSS_VS_LEMMA:g} points below LEMMA on every reader, and at least '
          f'x{MIN_SPEEDUP:g} faster than LEMMA on ' + ' and '.join(decision['speed_settings_checked']))
    print('   reading ' + ', '.join(f"{rec.upper()} {accuracy[rec]:.2f} (LEMMA {lemma[rec]:.2f}, "
                                    f"SGENet {decision['sgenet'][rec]})" for rec in exp.RECOGNIZERS)
          + f" -> {'ok' if decision['within_2_points_of_lemma'] else 'too far below LEMMA'}"
          + ('' if decision['reads_at_least_sgenet'] else ', and below SGENet'))
    print('   speed-up vs LEMMA ' + ', '.join(f'{key} x{value:.2f}' for key, value in speedup.items())
          + f" -> {'ok' if decision['at_least_x2_faster'] else 'too slow'}")
    print(f"   VERDICT: {decision['verdict']}")


def main():
    args = parse_args()
    results_path = os.path.join(args.out_dir, 'results.json')
    if os.path.exists(results_path):
        print(f'{results_path} exists: run already done')
        print_decision(exp.read_json(results_path)['decision'])
        return
    exp.set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = exp.load_config(args.config)
    textbase = exp.make_textbase(config, args.eval_batch_size)

    data = exp.load_data(config.TRAIN, args, os.path.join(args.out_dir, 'val_split.json'), with_test=True)
    print(f'1. split: {data.n_train} train / {data.n_val} validation images '
          '(LEMMA was trained on both; validation only reports progress)', flush=True)
    recognizers = {rec: textbase.load_recognizer(rec) for rec in exp.RECOGNIZERS}
    teacher = lemma_teacher.load_lemma(args.lemma, device)
    exp.set_seed(args.seed)  # reseed after loading: building the readers draws random numbers
    student = lemma_fast.LemmaFastSR(teacher, lemma_fast.GuidanceNet(args.width)).to(device)
    counts = lemma_fast.parameter_counts(student)
    print(f"   student: LEMMA's SR branch, frozen ({counts['sr_branch']:,} parameters) + GuidanceNet "
          f"width {args.width}, trained ({counts['guidance_branch']:,} parameters)")

    print(f'2. training GuidanceNet for {args.epochs} epochs (last epoch kept):', flush=True)
    history = train(student, teacher, data, args, device, textbase, recognizers['crnn'])
    model_path = os.path.join(args.out_dir, 'lemma_fast.pth')
    lemma_fast.save(student, model_path, lemma=args.lemma, width=args.width, epochs=args.epochs, seed=args.seed)
    difference = tsrn_pruning.max_output_difference(student, lemma_fast.load(model_path, device), device)
    print(f'   saved {model_path}; reload difference {difference:.1e}')
    if difference > 1e-4:
        raise RuntimeError(f'{model_path} does not reload to the same model')

    print('3. test accuracy with ASTER, MORAN and CRNN, and McNemar vs LEMMA...', flush=True)
    test = exp.evaluate_test(student, data.test, textbase, recognizers, device, keep_predictions=True)
    lemma_test = (exp.read_json(args.lemma_results)['test'] if args.lemma_results else
                  exp.evaluate_test(teacher, data.test, textbase, recognizers, device, keep_predictions=True))
    versus_lemma = compare_with_lemma(test, lemma_test)
    accuracy = test['average']['accuracy']
    for rec in exp.RECOGNIZERS:
        v = versus_lemma[rec]
        print(f"   {rec.upper():6s} {accuracy[rec]:.2f} (LEMMA {lemma_test['average']['accuracy'][rec]:.2f}, "
              f"{v['delta']:+.2f}, -{v['lost']}/+{v['gained']}, p={v['p']:.3g}; SGENet {SGENET[rec]})")

    print('4. size and speed (same session):', flush=True)
    size = {'student': {**counts, 'macs_aten': evaluate_lemma.aten_macs(student)},
            'LEMMA': {**lemma_teacher.parameter_counts(teacher), 'macs_aten': evaluate_lemma.aten_macs(teacher)}}
    for name, value in size.items():
        print(f'   {name:8s} ' + ', '.join(f'{k} {v:,}' for k, v in value.items() if v is not None))
    names = args.compare_names or [os.path.splitext(os.path.basename(c))[0] for c in args.compare]
    models = {'student': student, 'LEMMA': teacher,
              **{name: exp.load_tsrn(checkpoint, device) for name, checkpoint in zip(names, args.compare)}}
    latency = evaluate_lemma.time_models(models, args)
    for name, ms in latency.items():
        print(f'   {name:12s} ' + '  '.join(f"{key} {value:7.2f} ms (x{latency['LEMMA'][key] / value:.2f})"
                                          for key, value in ms.items()))
    print('   (xN = speed relative to LEMMA)')

    speedup = {key: latency['LEMMA'][key] / latency['student'][key] for key in latency['student']}
    decision = decide(accuracy, lemma_test['average']['accuracy'], speedup)
    exp.write_json(results_path, {'args': vars(args), 'environment': exp.environment_info(), 'history': history,
                                  'test': test, 'lemma_accuracy': lemma_test['average']['accuracy'],
                                  'versus_lemma': versus_lemma, 'size': size, 'latency_ms': latency,
                                  'decision': decision, 'gpu': exp.gpu_name()})
    print()
    print_decision(decision)
    print(f'saved {results_path}')


if __name__ == '__main__':
    main()
