"""Run LEMMA's released model in our pipeline: reading, size and speed, next to our TSRNs.

LEMMA (Guo et al., IJCAI 2023) is only run, never trained (lemma_teacher.py).
1. Check: every weight of the released file loads into the network (strict) and the output is a
   4 x 32 x 128 image.
2. Test accuracy with our three readers, per split, keeping what is read on every image. LEMMA's
   README reports ASTER / MORAN / CRNN 66.93 / 63.60 / 56.73 for this file, so this also checks
   our evaluation.
3. Size: parameters used at inference (guidance recognizer included) and MACs for one image.
4. Speed in one session, interleaved rounds, median: LEMMA, its guidance branch alone, and the TSRN
   checkpoints given with --compare (e.g. the full TSRN, then the 2-block T). Speed-ups are relative
   to the first --compare model. GPU batch 1 and 64, CPU 1 thread batch 1.

    python evaluate_lemma.py --lemma <LEMMA-release.pth> --compare <tsrn.pth> [...] --out_dir <folder>

Writes lemma_results.json (with what every reader read on every test image).
"""
import argparse
import copy
import os
import statistics

import torch

import lemma_teacher
import tsrn_experiment as exp
import tsrn_pruning


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lemma', required=True, help="LEMMA's released checkpoint")
    parser.add_argument('--compare', nargs='*', default=[], help='TSRN checkpoints to time in the same session')
    parser.add_argument('--compare_names', nargs='*', default=None, help='labels for --compare (default: file names)')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--config', default='config/super_resolution.yaml')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='MORAN supports at most 256')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--gpu_repeats', type=int, default=100)
    parser.add_argument('--cpu_repeats', type=int, default=20)
    args = parser.parse_args()
    if args.compare_names is not None and len(args.compare_names) != len(args.compare):
        parser.error('give one --compare_names label per --compare checkpoint')
    return args


def aten_macs(model):
    """MACs for one LR image counted by ptflops from the aten operations, so the matrix products inside
    attention are counted too (module hooks miss them). None if this ptflops cannot count it."""
    from ptflops import get_model_complexity_info
    try:
        macs, _ = get_model_complexity_info(copy.deepcopy(model).cpu().eval(), (4, *tsrn_pruning.LR_SIZE),
                                            as_strings=False, backend='aten', print_per_layer_stat=False,
                                            verbose=False)
        return int(macs)
    except Exception as error:  # an older ptflops or an operation it does not know
        print(f'   note: aten MAC count failed ({error})')
        return None


def time_models(models, args):
    """Median latency per model over interleaved rounds, so drift of the machine affects all alike."""
    timings = {name: [] for name in models}
    for round_index in range(1, args.rounds + 1):
        for name, model in models.items():
            timings[name].append(exp.measure_latency(model, gpu_repeats=args.gpu_repeats,
                                                     cpu_repeats=args.cpu_repeats))
        print(f'   round {round_index}/{args.rounds} done', flush=True)
    return {name: {key: statistics.median(t[key] for t in runs) for key in runs[0]} for name, runs in timings.items()}


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = exp.load_config(args.config)
    textbase = exp.make_textbase(config, args.eval_batch_size)

    print('1. loading LEMMA (strict: every weight must match)...', flush=True)
    lemma = lemma_teacher.load_lemma(args.lemma, device)
    with torch.no_grad():
        shape = tuple(lemma(torch.rand(2, 4, *tsrn_pruning.LR_SIZE, device=device)).shape)
    if shape != (2, 4, 32, 128):
        raise RuntimeError(f'LEMMA returned {shape}, expected (2, 4, 32, 128)')
    print(f'   ok, output {shape}')

    print('2. test accuracy with ASTER, MORAN and CRNN...', flush=True)
    recognizers = {rec: textbase.load_recognizer(rec) for rec in exp.RECOGNIZERS}
    test = exp.evaluate_test(lemma, exp.make_test_loaders(config.TRAIN, args.eval_batch_size), textbase,
                             recognizers, device, keep_predictions=True)
    accuracy = test['average']['accuracy']
    for split in exp.SPLIT_ORDER:
        print(f'   {split:6s} ' + ' / '.join(f"{test[split]['accuracy'][rec]:.2f}" for rec in exp.RECOGNIZERS)
              + f"  PSNR {test[split]['psnr']:.2f}")
    print('   average ASTER / MORAN / CRNN ' + ' / '.join(f'{accuracy[rec]:.2f}' for rec in exp.RECOGNIZERS)
          + '  (LEMMA README: ' + ' / '.join(f'{lemma_teacher.README_ACCURACY[rec]:.2f}' for rec in exp.RECOGNIZERS)
          + ')', flush=True)

    print('3. size:', flush=True)
    names = args.compare_names or [os.path.splitext(os.path.basename(c))[0] for c in args.compare]
    tsrns = {name: exp.load_tsrn(checkpoint, device) for name, checkpoint in zip(names, args.compare)}
    size = {'LEMMA': {**lemma_teacher.parameter_counts(lemma), 'macs_aten': aten_macs(lemma)}}
    for name, model in tsrns.items():  # TSRNs: both counts, so the aten count can be checked against hooks
        size[name] = {'inference': tsrn_pruning.inference_parameters(model),
                      'macs_hooks': tsrn_pruning.inference_macs(model), 'macs_aten': aten_macs(model)}
    for name, counts in size.items():
        print(f'   {name:12s} ' + ', '.join(f'{key} {value:,}' for key, value in counts.items() if value is not None))

    print(f'4. timing on {exp.gpu_name()} and 1 CPU thread, {args.rounds} interleaved rounds:', flush=True)
    models = {'LEMMA': lemma, 'LEMMA guidance only': lemma_teacher.LemmaGuidance(lemma), **tsrns}
    latency = time_models(models, args)
    reference = names[0] if names else 'LEMMA'
    for name, ms in latency.items():
        print(f'   {name:20s} ' + '  '.join(f"{key} {value:7.2f} ms (x{latency[reference][key] / value:.2f})"
                                          for key, value in ms.items()))
    print(f'   (xN = speed relative to {reference})')

    path = os.path.join(args.out_dir, 'lemma_results.json')
    exp.write_json(path, {'args': vars(args), 'environment': exp.environment_info(), 'test': test,
                          'readme_accuracy': lemma_teacher.README_ACCURACY, 'size': size,
                          'latency_ms': latency, 'speed_reference': reference, 'gpu': exp.gpu_name()})
    print(f'saved {path}')


if __name__ == '__main__':
    main()
