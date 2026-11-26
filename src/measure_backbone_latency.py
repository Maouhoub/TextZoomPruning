"""Speed-only test of cut-down LEMMA models (notebook sections 23 and 24). Random weights, no training.

Section 23: LEMMA's vision backbone (ResNet45 + 3 transformer layers) takes 78-87% of its guidance time
(section 22), so whole LEMMA models were timed with a thinner (`--widths`, ResNet45 and feed-forward width)
and shallower (`--layers`, transformer layers kept) backbone. No cut reached x2: the rest of LEMMA,
mainly its 5 GRU super-resolution blocks, is about half its time.

Section 24 (`--sr_blocks 5 3 2`): the same, with fewer of those SR blocks (each holds two GRUs that scan
the image step by step), combined with backbone cuts given by `--cuts`. A 2-block TSRN trained from
scratch kept its reading (section 19), so fewer SR blocks is the cut our own measurements point to.

Every model is timed in one session with the original LEMMA. Weights do not change the speed, so none are
loaded.

Rule fixed before the run: the candidate for fine-tuning is the largest model (the most MACs kept) that is
at least x2 faster than LEMMA on CPU batch 1 and on GPU batch 64. If none is, this line stops.

    python measure_backbone_latency.py --out_dir <folder>                                  # section 23
    python measure_backbone_latency.py --sr_blocks 5 3 2 --cuts L3_w1 L0_w0.35 --out_dir <folder>   # section 24

Writes backbone_latency.json.
"""
import argparse
import os
import re

import torch

import evaluate_lemma
import lemma_backbone
import lemma_teacher
import tsrn_experiment as exp

SPEED_SETTINGS = ('cpu1_b1', 'gpu_b64')
MIN_SPEEDUP = 2.0
LEMMA_SR_BLOCKS = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--layers', nargs='+', type=int, default=[3, 2, 1, 0], help='transformer layers kept')
    parser.add_argument('--widths', nargs='+', type=float, default=[1.0, 0.75, 0.5, 0.35],
                        help='ResNet45 and feed-forward width multipliers')
    parser.add_argument('--cuts', nargs='+', default=None,
                        help='backbone cuts such as L3_w1 L0_w0.35 (replaces the --layers x --widths grid)')
    parser.add_argument('--sr_blocks', nargs='+', type=int, default=[LEMMA_SR_BLOCKS],
                        help='SR blocks kept (LEMMA has 5)')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--gpu_repeats', type=int, default=30)
    parser.add_argument('--cpu_repeats', type=int, default=10)
    args = parser.parse_args()
    if args.cuts is None:
        args.cut_list = [(layers, width) for layers in args.layers for width in args.widths]
    else:
        args.cut_list = []
        for cut in args.cuts:
            match = re.fullmatch(r'L(\d+)_w([0-9.]+)', cut)
            if not match:
                parser.error(f'--cuts takes names like L3_w1 or L0_w0.35, not {cut}')
            args.cut_list.append((int(match.group(1)), float(match.group(2))))
    return args


def variant_name(blocks, layers, width):
    prefix = '' if blocks == LEMMA_SR_BLOCKS else f'B{blocks}_'
    return f'{prefix}L{layers}_w{width:g}'


def is_fast(speedup):
    """At least x2 faster than LEMMA on CPU batch 1 and GPU batch 64 (CPU only when there is no GPU)."""
    checked = [key for key in SPEED_SETTINGS if key in speedup]
    return bool(checked) and all(speedup[key] >= MIN_SPEEDUP for key in checked)


def build_models(args, device):
    torch.manual_seed(0)
    lemma = lemma_teacher.LemmaSR().to(device).eval()
    models, shapes = {'LEMMA': lemma}, {}
    for blocks in args.sr_blocks:
        base = lemma if blocks == LEMMA_SR_BLOCKS else lemma_teacher.LemmaSR(srb_nums=blocks).to(device).eval()
        for layers, width in args.cut_list:
            full_backbone = layers == lemma_backbone.LAYERS and width == 1.0
            if full_backbone and blocks == LEMMA_SR_BLOCKS:
                continue  # that is LEMMA itself
            name = variant_name(blocks, layers, width)
            if full_backbone:
                models[name] = base
                shapes[name] = {'sr_blocks': blocks, 'layers': layers, 'width': width}
                continue
            backbone = lemma_backbone.SlimBackbone(width, layers)
            models[name] = lemma_backbone.with_backbone(base, backbone)
            shapes[name] = {'sr_blocks': blocks, 'layers': layers, 'width': width, 'resnet_outs': backbone.resnet.outs,
                            'resnet_mids': backbone.resnet.mids, 'ffn': backbone.ffn}
    return models, shapes


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, shapes = build_models(args, device)

    print(f'1. size of {len(models)} models (whole LEMMA with fewer SR blocks and/or a cut backbone)...', flush=True)
    size = {name: {'inference': lemma_teacher.parameter_counts(model)['inference'],
                   'backbone': lemma_backbone.backbone_parameters(model),
                   'macs_aten': evaluate_lemma.aten_macs(model)} for name, model in models.items()}
    print(f'2. timing on {exp.gpu_name()} and 1 CPU thread, {args.rounds} interleaved rounds...', flush=True)
    latency = evaluate_lemma.time_models(models, args)
    speedup = {name: {key: latency['LEMMA'][key] / ms[key] for key in ms} for name, ms in latency.items()}
    passes = {name: is_fast(speedup[name]) for name in models}

    print()
    print(f"   {'model':14s} {'params':>11s} {'backbone':>11s} {'GMACs':>6s}  "
          + '  '.join(f'{key:>17s}' for key in latency['LEMMA']) + '  x2 on CPU b1 and GPU b64')
    # Largest model first (the mildest cut).
    order = sorted(models, key=lambda n: (size[n]['macs_aten'] or 0, size[n]['inference']), reverse=True)
    for name in order:
        macs = size[name]['macs_aten']
        print(f"   {name:14s} {size[name]['inference']:>11,} {size[name]['backbone']:>11,} "
              f"{'' if macs is None else f'{macs / 1e9:6.2f}'}  "
              + '  '.join(f"{latency[name][key]:7.2f} ms (x{speedup[name][key]:.2f})" for key in latency['LEMMA'])
              + ('  yes' if passes[name] and name != 'LEMMA' else ''))
    candidates = [name for name in order if passes[name] and name != 'LEMMA']
    choice = candidates[0] if candidates else None
    print()
    if choice:
        print(f'rule: the largest model that is x{MIN_SPEEDUP:g} faster on CPU b1 and GPU b64 -> {choice} '
              f"({size[choice]['inference']:,} parameters, vs LEMMA {size['LEMMA']['inference']:,})")
    else:
        print(f'rule: no model reaches x{MIN_SPEEDUP:g} on CPU b1 and GPU b64 -> this line stops here')
    path = os.path.join(args.out_dir, 'backbone_latency.json')
    exp.write_json(path, {'args': {k: v for k, v in vars(args).items() if k != 'cut_list'},
                          'environment': exp.environment_info(), 'gpu': exp.gpu_name(), 'shapes': shapes,
                          'size': size, 'latency_ms': latency, 'speedup_vs_lemma': speedup, 'passes_x2': passes,
                          'choice': choice})
    print(f'saved {path}')


if __name__ == '__main__':
    main()
