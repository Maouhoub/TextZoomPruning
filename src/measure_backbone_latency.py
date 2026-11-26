"""Speed-only test of pruned LEMMA backbones (notebook section 23, step 1). Random weights, no training.

LEMMA's vision backbone (ResNet45 + 3 transformer layers) takes 78-87% of its guidance time (section 22),
and everything after it must stay as released. This test times whole LEMMA models whose backbone is
thinner (`--widths`, ResNet45 width and transformer feed-forward width) and shallower (`--layers`,
transformer layers kept), in one session with the original LEMMA. Weights do not change the speed, so no
weights are loaded.

Rule fixed before the run: the candidate for step 2 (pruning from LEMMA's weights, then fine-tuning) is
the mildest cut (the most backbone MACs kept) that makes the whole LEMMA at least x2 faster on CPU batch 1
and on GPU batch 64. If no cut does, the backbone line stops here.

    python measure_backbone_latency.py --out_dir <folder>

Writes backbone_latency.json.
"""
import argparse
import os

import torch

import evaluate_lemma
import lemma_backbone
import lemma_teacher
import tsrn_experiment as exp

SPEED_SETTINGS = ('cpu1_b1', 'gpu_b64')
MIN_SPEEDUP = 2.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--layers', nargs='+', type=int, default=[3, 2, 1, 0], help='transformer layers kept')
    parser.add_argument('--widths', nargs='+', type=float, default=[1.0, 0.75, 0.5, 0.35],
                        help='ResNet45 and feed-forward width multipliers')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--gpu_repeats', type=int, default=30)
    parser.add_argument('--cpu_repeats', type=int, default=10)
    return parser.parse_args()


def variant_name(layers, width):
    return f'L{layers}_w{width:g}'


def is_fast(speedup):
    """At least x2 faster than LEMMA on CPU batch 1 and GPU batch 64 (CPU only when there is no GPU)."""
    checked = [key for key in SPEED_SETTINGS if key in speedup]
    return bool(checked) and all(speedup[key] >= MIN_SPEEDUP for key in checked)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    lemma = lemma_teacher.LemmaSR().to(device).eval()
    models, shapes = {'LEMMA': lemma}, {}
    for layers in args.layers:
        for width in args.widths:
            if layers == lemma_backbone.LAYERS and width == 1.0:
                continue  # that is LEMMA itself
            backbone = lemma_backbone.SlimBackbone(width, layers)
            models[variant_name(layers, width)] = lemma_backbone.with_backbone(lemma, backbone)
            shapes[variant_name(layers, width)] = {'layers': layers, 'width': width, 'resnet_outs': backbone.resnet.outs,
                                                   'resnet_mids': backbone.resnet.mids, 'ffn': backbone.ffn}

    print(f'1. size of {len(models)} models (whole LEMMA, the backbone changed)...', flush=True)
    size = {name: {'inference': lemma_teacher.parameter_counts(model)['inference'],
                   'backbone': lemma_backbone.backbone_parameters(model),
                   'macs_aten': evaluate_lemma.aten_macs(model)} for name, model in models.items()}
    print(f'2. timing on {exp.gpu_name()} and 1 CPU thread, {args.rounds} interleaved rounds...', flush=True)
    latency = evaluate_lemma.time_models(models, args)
    speedup = {name: {key: latency['LEMMA'][key] / ms[key] for key in ms} for name, ms in latency.items()}
    passes = {name: is_fast(speedup[name]) for name in models}

    print()
    print(f"   {'model':10s} {'params':>11s} {'backbone':>11s} {'GMACs':>6s}  "
          + '  '.join(f'{key:>17s}' for key in latency['LEMMA']) + '  x2 on CPU b1 and GPU b64')
    # Mildest cut first: the other parts are the same in every model, so more MACs = more backbone kept.
    order = sorted(models, key=lambda n: (size[n]['macs_aten'] or 0, size[n]['backbone']), reverse=True)
    for name in order:
        macs = size[name]['macs_aten']
        print(f"   {name:10s} {size[name]['inference']:>11,} {size[name]['backbone']:>11,} "
              f"{'' if macs is None else f'{macs / 1e9:6.2f}'}  "
              + '  '.join(f"{latency[name][key]:7.2f} ms (x{speedup[name][key]:.2f})" for key in latency['LEMMA'])
              + ('  yes' if passes[name] and name != 'LEMMA' else ''))
    candidates = [name for name in order if passes[name] and name != 'LEMMA']
    choice = candidates[0] if candidates else None
    print()
    if choice:
        print(f'rule: the mildest cut that is x{MIN_SPEEDUP:g} faster on CPU b1 and GPU b64 -> {choice} '
              f"({size[choice]['backbone']:,} backbone parameters, vs {size['LEMMA']['backbone']:,})")
    else:
        print(f'rule: no cut reaches x{MIN_SPEEDUP:g} on CPU b1 and GPU b64 -> the backbone line stops here')
    path = os.path.join(args.out_dir, 'backbone_latency.json')
    exp.write_json(path, {'args': vars(args), 'environment': exp.environment_info(), 'gpu': exp.gpu_name(),
                          'shapes': shapes, 'size': size, 'latency_ms': latency, 'speedup_vs_lemma': speedup,
                          'passes_x2': passes, 'choice': choice})
    print(f'saved {path}')


if __name__ == '__main__':
    main()
