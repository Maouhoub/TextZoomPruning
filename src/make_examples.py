"""Example figure: what pruning does to the image and to what a reader reads.

Picks test images that the unpruned model gets right and the most pruned step gets wrong (for
`--reader`), a few per split, and shows for each one: the low-resolution input, the output of
every chosen step with what the reader reads there and the image's PSNR, and the high-resolution
target. The readings come from rescore/ (run rescore_trajectory.py first); the images are made
again with the saved models.

    python make_examples.py --run_dir <trajectory run folder> [--steps 0 4 7] [--per_split 2]

Writes figures/examples_<reader>.png and .pdf, and examples_<reader>.json, in the run folder.
"""
import argparse
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import tsrn_experiment as exp
from dataset.dataset import alignCollate_real, lmdbDataset_real
from utils.ssim_psnr import calculate_psnr

RIGHT, WRONG = '#1b7f3b', '#c0392b'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--config', default='config/super_resolution.yaml')
    parser.add_argument('--steps', type=int, nargs='+', default=[0, 4, 7], help='steps to show, first = reference')
    parser.add_argument('--reader', default='crnn', choices=exp.RECOGNIZERS)
    parser.add_argument('--per_split', type=int, default=2, help='examples per test split')
    parser.add_argument('--seed', type=int, default=0, help='which candidates are drawn')
    return parser.parse_args()


def choose(rescore, labels, steps, reader, per_split, seed):
    """Images right at the first step and wrong at the last one; middle steps right first if possible."""
    rng = random.Random(seed)
    chosen = []
    for split in exp.SPLIT_ORDER:
        def right(step, i):
            return rescore[step]['sets'][split]['predictions'][reader][i] == labels[split][i]
        candidates = [i for i in range(len(labels[split])) if right(steps[0], i) and not right(steps[-1], i)]
        middle_right = [i for i in candidates if all(right(s, i) for s in steps[1:-1])]
        pool = middle_right if len(middle_right) >= per_split else candidates
        chosen += [(split, i) for i in sorted(rng.sample(pool, min(per_split, len(pool))))]
    return chosen


def load_images(config, chosen):
    cfg = config.TRAIN
    by_name = {os.path.basename(os.path.normpath(d)): d for d in cfg.VAL.val_data_dir}
    datasets = {split: lmdbDataset_real(root=by_name[split], voc_type=cfg.voc_type, max_len=cfg.max_len, test=True)
                for split in exp.SPLIT_ORDER}
    collate = alignCollate_real(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale, mask=True)
    return collate([datasets[split][i] for split, i in chosen])


def to_rgb(image):
    return image[:3].clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = exp.load_config(args.config)
    rescore_dir = os.path.join(args.run_dir, 'rescore')
    labels = exp.read_json(os.path.join(rescore_dir, 'labels.json'))
    rescore = {s: exp.read_json(os.path.join(rescore_dir, f'step{s:02d}.json')) for s in args.steps}
    chosen = choose(rescore, labels, args.steps, args.reader, args.per_split, args.seed)
    if not chosen:
        raise SystemExit(f'no test image is read right at step {args.steps[0]} and wrong at step {args.steps[-1]}')
    images_hr, images_lr, _ = load_images(config, chosen)

    outputs = {}
    for s in args.steps:
        checkpoint = rescore[s]['checkpoint']
        model = exp.load_tsrn(checkpoint, device) if s == 0 else exp.load_pruned(checkpoint, device)
        with torch.no_grad():
            outputs[s] = model.eval()(images_lr.to(device)).cpu()

    columns = ['input (LR, bicubic ×2)'] + [f"{rescore[s]['removed_pct']:g}% removed" for s in args.steps] + ['target (HR)']
    fig, axes = plt.subplots(len(chosen), len(columns), figsize=(2.6 * len(columns), 1.25 * len(chosen) + 0.4),
                             squeeze=False)
    lr_up = F.interpolate(images_lr, size=images_hr.shape[-2:], mode='bicubic', align_corners=False)
    record = []
    for row, (split, i) in enumerate(chosen):
        label = labels[split][i]
        panels = [(to_rgb(lr_up[row]), f'{split}, "{label}"', 'black')]
        readings = {}
        for s in args.steps:
            read = rescore[s]['sets'][split]['predictions'][args.reader][i]
            psnr = calculate_psnr(outputs[s][row:row + 1], images_hr[row:row + 1]).item()
            readings[s] = {'read': read, 'psnr': psnr}
            panels.append((to_rgb(outputs[s][row]), f'"{read}"  {psnr:.1f} dB', RIGHT if read == label else WRONG))
        panels.append((to_rgb(images_hr[row]), '', 'black'))
        for col, (image, caption, color) in enumerate(panels):
            ax = axes[row][col]
            ax.imshow(image, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(caption, color=color, fontsize=9)
            if row == 0:
                ax.set_title(columns[col], fontsize=9)
        record.append({'split': split, 'index': i, 'label': label, 'steps': readings})

    fig.suptitle(f'{args.reader.upper()} readings (green = right, red = wrong) and PSNR of each output', fontsize=10)
    fig.tight_layout()
    out = os.path.join(args.run_dir, 'figures', f'examples_{args.reader}')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=200)
    exp.write_json(f'{out}.json', record)
    print(f'saved {out}.png, .pdf and .json')


if __name__ == '__main__':
    main()
