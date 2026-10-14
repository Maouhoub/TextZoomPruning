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
import copy
import json
import os
import random
import statistics
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader, Subset

import tsrn_pruning
from dataset.dataset import ConcatDataset, alignCollate_real, lmdbDataset_real
from interfaces.base import TextBase
from loss.image_loss import ImageLoss
from model.tsrn import TSRN
from profile_tsrn import median_latency_ms
from utils.util import str_filt

RECOGNIZERS = ('aster', 'moran', 'crnn')
SPLIT_ORDER = ('easy', 'medium', 'hard')


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tsrn(checkpoint, device):
    # Same configuration as training with --STN --mask (5 SRBs, 32 GRU hidden units).
    model = TSRN(scale_factor=2, width=128, height=32, STN=True, mask=True, srb_nums=5, hidden_units=32)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict_G'])
    return model.to(device)


def make_loader(data, batch_size, cfg, shuffle):
    collate = alignCollate_real(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale, mask=True)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=int(cfg.workers),
                      collate_fn=collate, drop_last=shuffle)


@torch.no_grad()
def evaluate(model, loader, textbase, recognizers, device):
    """Accuracy (%) per recognizer, plus PSNR/SSIM with the TextZoom protocol (per batch, then averaged)."""
    model.eval()
    correct = dict.fromkeys(recognizers, 0)
    psnr, ssim, total = [], [], 0
    for images_hr, images_lr, labels in loader:
        images_sr = model(images_lr.to(device))
        images_hr = images_hr.to(device)
        psnr.append(textbase.cal_psnr(images_sr, images_hr).item())
        ssim.append(textbase.cal_ssim(images_sr, images_hr).item())
        targets = [str_filt(label, 'lower') for label in labels]
        for name, (recognizer, aster_info) in recognizers.items():
            predictions = textbase.recognize(name, recognizer, images_sr, aster_info)
            correct[name] += sum(str_filt(p, 'lower') == t for p, t in zip(predictions, targets))
        total += len(labels)
    return {'accuracy': {name: 100.0 * c / total for name, c in correct.items()},
            'psnr': statistics.mean(psnr), 'ssim': statistics.mean(ssim), 'images': total}


def evaluate_test(model, test_loaders, textbase, recognizers, device):
    results = {name: evaluate(model, loader, textbase, recognizers, device) for name, loader in test_loaders.items()}
    images = sum(r['images'] for r in results.values())
    results['average'] = {'accuracy': {rec: sum(r['accuracy'][rec] * r['images'] for r in results.values()) / images
                                       for rec in recognizers}}
    return results


def measure_latency(model):
    """Median latency (ms): GPU at batch 1 and 64, CPU with 1 thread at batch 1."""
    latency = {}
    if torch.cuda.is_available():
        gpu_model = model.eval()
        for batch_size in (1, 64):
            x = torch.rand(batch_size, 4, *tsrn_pruning.LR_SIZE, device='cuda')
            latency[f'gpu_b{batch_size}'] = median_latency_ms(gpu_model, x, warmup=10, repeats=50)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_model = copy.deepcopy(model).cpu().eval()
    latency['cpu1_b1'] = median_latency_ms(cpu_model, torch.rand(1, 4, *tsrn_pruning.LR_SIZE), warmup=3, repeats=20)
    torch.set_num_threads(threads)
    return latency


def describe(model, device):
    return {'params': tsrn_pruning.inference_parameters(model),
            'macs': tsrn_pruning.inference_macs(model),
            'latency_ms': measure_latency(model.to(device))}


def fine_tune(student, teacher, train_loader, val_loader, textbase, val_recognizers, args, device):
    """Train with the original TSRN loss plus output distillation; keep the best epoch on validation.

    Teacher and student see the same STN-aligned input, so the distillation target matches.
    """
    criterion = ImageLoss(gradient=True, loss_weight=[1, 1e-4]).to(device)
    optimizer = optim.Adam(student.parameters(), lr=args.lr, betas=(0.5, 0.999))
    best_score, best_state, history = -float('inf'), None, []
    for epoch in range(1, args.epochs + 1):
        student.train()
        losses = []
        for images_hr, images_lr, _ in train_loader:
            images_lr, images_hr = images_lr.to(device), images_hr.to(device)
            aligned = student.align(images_lr)
            images_sr = student.reconstruct(aligned)
            loss = criterion(images_sr, images_hr)
            if args.kd_weight > 0:
                with torch.no_grad():
                    target = teacher.reconstruct(aligned.detach())
                loss = loss + args.kd_weight * F.mse_loss(images_sr, target)
            loss = loss * 100  # same scaling as the original TSRN training loop
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 0.25)
            optimizer.step()
            losses.append(loss.item())

        val = evaluate(student, val_loader, textbase, val_recognizers, device)
        score = val['psnr'] if args.select_by == 'psnr' else val['accuracy']['crnn']
        history.append({'epoch': epoch, 'loss': statistics.mean(losses), 'val': val})
        print(f"  epoch {epoch}: loss {statistics.mean(losses):.4f} | val CRNN {val['accuracy']['crnn']:.2f}% "
              f"PSNR {val['psnr']:.2f}", flush=True)
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(student.state_dict())
    student.load_state_dict(best_state)
    return history


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
              + ''.join(cell(test.get(rec), '.1f', 7) for rec in RECOGNIZERS)
              + ''.join(cell(latency.get(key), '.1f', 8) for key in ('gpu_b1', 'gpu_b64', 'cpu1_b1')))
    print('Accuracy in %, test = weighted average over easy/medium/hard, latency in ms per batch.')


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = EasyDict(yaml.load(open(args.config, 'r'), Loader=yaml.Loader))
    cfg = config.TRAIN
    textbase = TextBase(config, SimpleNamespace(syn=False, mixed=False, resume=None, mask=True, vis_dir=None,
                                                batch_size=args.eval_batch_size, test_data_dir=None))

    # 1. Validation split of the training set.
    train_full = ConcatDataset([lmdbDataset_real(root=d, voc_type=cfg.voc_type, max_len=cfg.max_len)
                                for d in cfg.train_data_dir])
    train_idx, val_idx = tsrn_pruning.split_indices(len(train_full), args.val_fraction, args.seed)
    with open(os.path.join(args.out_dir, 'val_split.json'), 'w') as f:
        json.dump({'train_data_dir': list(cfg.train_data_dir), 'n_total': len(train_full),
                   'val_fraction': args.val_fraction, 'seed': args.seed, 'val_indices': val_idx}, f)
    print(f'1. split: {len(train_idx)} train / {len(val_idx)} validation images', flush=True)
    train_loader = make_loader(Subset(train_full, train_idx), args.batch_size, cfg, shuffle=True)
    val_loader = make_loader(Subset(train_full, val_idx), args.eval_batch_size, cfg, shuffle=False)
    test_loaders = {}
    if not args.skip_test:
        by_name = {os.path.basename(os.path.normpath(d)): d for d in cfg.VAL.val_data_dir}
        test_loaders = {name: make_loader(lmdbDataset_real(root=by_name[name], voc_type=cfg.voc_type,
                                                           max_len=cfg.max_len, test=True),
                                          args.eval_batch_size, cfg, shuffle=False)
                        for name in SPLIT_ORDER}

    recognizers = {rec: textbase.load_recognizer(rec) for rec in (RECOGNIZERS if test_loaders else ('crnn',))}
    val_recognizers = {'crnn': recognizers['crnn']}

    student = load_tsrn(args.resume, device)
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    stages = {}
    print('   measuring the unpruned model...', flush=True)
    stages['baseline'] = describe(student, device)
    stages['baseline']['val'] = evaluate(student, val_loader, textbase, val_recognizers, device)
    if test_loaders:
        stages['baseline']['test'] = evaluate_test(student, test_loaders, textbase, recognizers, device)

    # 2. One pruning step on the convolutions.
    criterion = ImageLoss(gradient=True, loss_weight=[1, 1e-4]).to(device)
    if args.importance == 'taylor':
        tsrn_pruning.accumulate_gradients(student, train_loader, criterion, args.taylor_batches, device)
    report = tsrn_pruning.prune(student, args.ratio, args.importance, args.round_to)
    print(f'2. pruned {args.ratio:.0%} per group ({args.importance}):')
    for name, (before, after) in report.items():
        print(f'     {name:16s} {before} -> {after}')

    # 3. Structural check.
    print(f'3. output shape after pruning: {tsrn_pruning.check_structure(student, device)}', flush=True)
    stages['pruned, no fine-tune'] = {'params': tsrn_pruning.inference_parameters(student),
                                      'macs': tsrn_pruning.inference_macs(student),
                                      'val': evaluate(student, val_loader, textbase, val_recognizers, device)}

    # 4. Short fine-tune with the unpruned model as teacher.
    print(f'4. fine-tuning for {args.epochs} epoch(s), best epoch by validation {args.select_by}:', flush=True)
    history = fine_tune(student, teacher, train_loader, val_loader, textbase, val_recognizers, args, device)

    # 5. Measure the pruned model.
    print('5. measuring the pruned model...', flush=True)
    stages['pruned + fine-tune'] = describe(student, device)
    stages['pruned + fine-tune']['val'] = evaluate(student, val_loader, textbase, val_recognizers, device)
    if test_loaders:
        stages['pruned + fine-tune']['test'] = evaluate_test(student, test_loaders, textbase, recognizers, device)

    torch.save({'state_dict_G': student.state_dict(), 'channels': tsrn_pruning.channel_sizes(student),
                'info': {'source': args.resume, 'ratio': args.ratio, 'importance': args.importance}},
               os.path.join(args.out_dir, 'tsrn_pruned.pth'))
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump({'args': vars(args), 'pruning': report, 'fine_tune': history, 'stages': stages}, f, indent=2)
    print()
    print_summary(stages)
    print(f'\nSaved results.json, val_split.json and tsrn_pruned.pth to {args.out_dir}')


if __name__ == '__main__':
    main()
