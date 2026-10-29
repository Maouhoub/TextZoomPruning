"""Shared pieces of the TSRN pruning experiments: data, evaluation, fine-tuning, latency, storage."""
import copy
import json
import os
import platform
import random
import statistics
import subprocess
import time
from importlib import metadata
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader, RandomSampler, Subset

import tsrn_pruning
from dataset.dataset import ConcatDataset, alignCollate_real, lmdbDataset_real
from interfaces.base import TextBase
from loss.image_loss import ImageLoss
from loss.recognition_loss import CRNNRecognitionLoss
from model.tsrn import TSRN
from profile_tsrn import median_latency_ms
from utils.util import str_filt

RECOGNIZERS = ('aster', 'moran', 'crnn')
SPLIT_ORDER = ('easy', 'medium', 'hard')


# ---------------------------------------------------------------- setup

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.load(f, Loader=yaml.Loader))


def make_textbase(config, eval_batch_size):
    """TextBase gives access to the recognizers and the TextZoom PSNR/SSIM, exactly as main.py uses them."""
    return TextBase(config, SimpleNamespace(syn=False, mixed=False, resume=None, mask=True, vis_dir=None,
                                            batch_size=eval_batch_size, test_data_dir=None))


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return 'unknown'


def environment_info():
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)),
                            capture_output=True, text=True).stdout.strip()
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'torchvision': package_version('torchvision'), 'torch_pruning': package_version('torch-pruning'),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'gpu': gpu_name(),
            'cpu_cores': os.cpu_count(), 'git_commit': commit}


# ---------------------------------------------------------------- models

def build_tsrn():
    # Same configuration as training with --STN --mask (5 SRBs, 32 GRU hidden units).
    return TSRN(scale_factor=2, width=128, height=32, STN=True, mask=True, srb_nums=5, hidden_units=32)


def load_tsrn(checkpoint, device):
    model = build_tsrn()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict_G'])
    return tsrn_pruning.compact_gru_weights(model.to(device))


def load_pruned(checkpoint, device):
    """A pruned TSRN saved with save_pruned: the channel sizes are restored before the weights."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = tsrn_pruning.shrink_to(build_tsrn(), saved['channels'])
    model.load_state_dict(saved['state_dict_G'])
    return tsrn_pruning.compact_gru_weights(model.to(device))


def save_pruned(model, path, **info):
    save_atomic({'state_dict_G': model.state_dict(), 'channels': tsrn_pruning.channel_sizes(model), 'info': info},
                path)


def frozen_copy(model):
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return tsrn_pruning.compact_gru_weights(teacher)


def checkpoint_info(checkpoint):
    return torch.load(checkpoint, map_location='cpu', weights_only=True).get('info', {})


def baseline_name(checkpoint):
    """Short name of an unpruned TSRN: the one stored by train_tsrn_baseline.py, else the file name
    without 'tsrn_' (the community checkpoint is 'community')."""
    stem = os.path.splitext(os.path.basename(checkpoint))[0]
    return checkpoint_info(checkpoint).get('name') or stem.removeprefix('tsrn_')


def check_baseline_split(checkpoint, val_indices, log=print):
    """Refuse a baseline trained on another split: its training images would include our validation images."""
    split = checkpoint_info(checkpoint).get('val_split')
    if split is None:
        log('WARNING: the unpruned checkpoint does not record its training split, so its training data may '
            'include the validation images (true for the community checkpoint)')
    elif split['val_indices'] != val_indices:
        raise SystemExit(f'{checkpoint} was trained with another validation split; '
                         'use the same --val_fraction and --split_seed as its training')


# ---------------------------------------------------------------- data

def make_loader(data, batch_size, cfg, shuffle, generator=None, persistent=False):
    """With `generator`, only that generator drives the shuffling, so reseeding it fixes the batch
    order. `persistent` keeps the loading processes alive between epochs."""
    collate = alignCollate_real(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale, mask=True)
    workers = int(cfg.workers)
    sampler = RandomSampler(data, generator=generator) if shuffle else None  # same as shuffle=True
    return DataLoader(data, batch_size=batch_size, sampler=sampler, num_workers=workers,
                      collate_fn=collate, drop_last=shuffle, persistent_workers=persistent and workers > 0)


def load_data(cfg, args, split_path, with_test):
    """Train/validation split of the TextZoom training set, and the three test splits.

    The split depends only on `args.split_seed` (not on the run seed), so every run and the clean
    baseline share the same validation images. It is written to `split_path` the first time and
    checked against it afterwards, so a resumed run can never silently use a different validation set.
    """
    train_full = ConcatDataset([lmdbDataset_real(root=d, voc_type=cfg.voc_type, max_len=cfg.max_len)
                                for d in cfg.train_data_dir])
    train_idx, val_idx = tsrn_pruning.split_indices(len(train_full), args.val_fraction, args.split_seed)
    split = {'train_data': [os.path.basename(os.path.normpath(d)) for d in cfg.train_data_dir],
             'n_total': len(train_full), 'val_fraction': args.val_fraction, 'seed': args.split_seed,
             'val_indices': val_idx}
    if os.path.exists(split_path):
        saved = read_json(split_path)
        if saved['n_total'] != split['n_total'] or saved['val_indices'] != val_idx:
            raise RuntimeError(f'the validation split differs from the one saved in {split_path}')
    else:
        write_json(split_path, split)

    test = {}
    if with_test:
        by_name = {os.path.basename(os.path.normpath(d)): d for d in cfg.VAL.val_data_dir}
        test = {name: make_loader(lmdbDataset_real(root=by_name[name], voc_type=cfg.voc_type,
                                                   max_len=cfg.max_len, test=True),
                                  args.eval_batch_size, cfg, shuffle=False)
                for name in SPLIT_ORDER}
    return SimpleNamespace(train=make_loader(Subset(train_full, train_idx), args.batch_size, cfg, shuffle=True),
                           val=make_loader(Subset(train_full, val_idx), args.eval_batch_size, cfg, shuffle=False),
                           test=test, n_train=len(train_idx), n_val=len(val_idx), val_indices=val_idx)


# ---------------------------------------------------------------- measurements

@torch.no_grad()
def predict(model, loader, textbase, recognizers, device):
    """What every recognizer reads on every image (lower case, as compared with the labels), the
    labels, and PSNR/SSIM with the TextZoom protocol (per batch, then averaged)."""
    model.eval()
    predictions = {name: [] for name in recognizers}
    labels, psnr, ssim = [], [], []
    for images_hr, images_lr, batch_labels in loader:
        images_sr = model(images_lr.to(device))
        images_hr = images_hr.to(device)
        psnr.append(textbase.cal_psnr(images_sr, images_hr).item())
        ssim.append(textbase.cal_ssim(images_sr, images_hr).item())
        labels += [str_filt(label, 'lower') for label in batch_labels]
        for name, (recognizer, aster_info) in recognizers.items():
            predictions[name] += [str_filt(p, 'lower')
                                  for p in textbase.recognize(name, recognizer, images_sr, aster_info)]
    return {'predictions': predictions, 'labels': labels, 'psnr': statistics.mean(psnr), 'ssim': statistics.mean(ssim)}


def summarize(result):
    """Accuracy (%) per recognizer and PSNR/SSIM from the output of `predict`."""
    labels = result['labels']
    return {'accuracy': {name: 100.0 * sum(p == t for p, t in zip(read, labels)) / len(labels)
                         for name, read in result['predictions'].items()},
            'psnr': result['psnr'], 'ssim': result['ssim'], 'images': len(labels)}


def evaluate(model, loader, textbase, recognizers, device):
    """Accuracy (%) per recognizer, plus PSNR/SSIM with the TextZoom protocol (per batch, then averaged)."""
    return summarize(predict(model, loader, textbase, recognizers, device))


def evaluate_test(model, test_loaders, textbase, recognizers, device, keep_predictions=False):
    """Per split and weighted average; with `keep_predictions`, also what every reader read on every image."""
    predicted = {name: predict(model, loader, textbase, recognizers, device) for name, loader in test_loaders.items()}
    results = {name: summarize(p) for name, p in predicted.items()}
    images = sum(r['images'] for r in results.values())
    results['average'] = {'accuracy': {rec: sum(r['accuracy'][rec] * r['images'] for r in results.values()) / images
                                       for rec in recognizers}}
    if keep_predictions:
        results['predictions'] = {name: {'labels': p['labels'], 'read': p['predictions']} for name, p in predicted.items()}
    return results


def measure_latency(model, gpu_repeats=50, cpu_repeats=20):
    """Median latency (ms): GPU at batch 1 and 64, CPU with 1 thread at batch 1."""
    latency = {}
    if torch.cuda.is_available():
        gpu_model = tsrn_pruning.compact_gru_weights(model.cuda().eval())
        for batch_size in (1, 64):
            x = torch.rand(batch_size, 4, *tsrn_pruning.LR_SIZE, device='cuda')
            latency[f'gpu_b{batch_size}'] = median_latency_ms(gpu_model, x, warmup=10, repeats=gpu_repeats)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_model = copy.deepcopy(model).cpu().eval()
    latency['cpu1_b1'] = median_latency_ms(cpu_model, torch.rand(1, 4, *tsrn_pruning.LR_SIZE), warmup=3,
                                           repeats=cpu_repeats)
    torch.set_num_threads(threads)
    return latency


def describe(model, device):
    return {'params': tsrn_pruning.inference_parameters(model),
            'macs': tsrn_pruning.inference_macs(model),
            'latency_ms': measure_latency(model.to(device)),
            'gpu': gpu_name()}


# ---------------------------------------------------------------- objectives

def make_recognition_loss(textbase):
    """CTC loss of its own frozen CRNN (separate from the CRNN used for evaluation)."""
    crnn, _ = textbase.load_recognizer('crnn')
    return CRNNRecognitionLoss(crnn, textbase.converter_crnn)


def scoring_loss(image_criterion, recognition=None, rec_weight=0.0):
    """Loss whose gradients give the Taylor importance: the image loss, plus the reading loss if weighted.

    With rec_weight 0 it is exactly the image loss used before. Otherwise both terms use the
    fine-tuning scale: 100 × image loss + rec_weight × CTC loss.
    """
    if rec_weight == 0:
        return lambda images_sr, images_hr, labels: image_criterion(images_sr, images_hr)
    return lambda images_sr, images_hr, labels: (100 * image_criterion(images_sr, images_hr)
                                                 + rec_weight * recognition(images_sr, labels))


# ---------------------------------------------------------------- fine-tuning

def fine_tune(student, teacher, train_loader, val_loader, textbase, val_recognizers, args, device, log=print,
              recognition=None):
    """Train with the original TSRN loss plus output distillation; keep the best epoch on validation.

    Runs at most `args.epochs` epochs. With `args.patience` > 0 it stops early once the validation
    metric (`args.select_by`) has not improved by more than `args.min_delta` for `patience` epochs,
    as in Chapter 1. Teacher and student see the same STN-aligned input, so the distillation target
    matches. With `args.ft_rec_weight` > 0, the CTC loss of a frozen CRNN reading the output
    (`recognition`) is added with that weight. Returns (history, best epoch).
    """
    criterion = ImageLoss(gradient=True, loss_weight=[1, 1e-4]).to(device)
    rec_weight = getattr(args, 'ft_rec_weight', 0.0)
    optimizer = optim.Adam(student.parameters(), lr=args.lr, betas=(0.5, 0.999))
    best_score, best_state, best_epoch, history = -float('inf'), None, None, []
    epochs_without_gain = 0
    for epoch in range(1, args.epochs + 1):
        student.train()
        losses = []
        for images_hr, images_lr, labels in train_loader:
            images_lr, images_hr = images_lr.to(device), images_hr.to(device)
            aligned = student.align(images_lr)
            images_sr = student.reconstruct(aligned)
            loss = criterion(images_sr, images_hr)
            if args.kd_weight > 0:
                with torch.no_grad():
                    target = teacher.reconstruct(aligned.detach())
                loss = loss + args.kd_weight * F.mse_loss(images_sr, target)
            loss = loss * 100  # same scaling as the original TSRN training loop
            if rec_weight > 0:
                loss = loss + rec_weight * recognition(images_sr, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 0.25)
            optimizer.step()
            losses.append(loss.item())

        val = evaluate(student, val_loader, textbase, val_recognizers, device)
        score = val['psnr'] if args.select_by == 'psnr' else val['accuracy']['crnn']
        history.append({'epoch': epoch, 'loss': statistics.mean(losses), 'val': val})
        log(f"  epoch {epoch}: loss {statistics.mean(losses):.4f} | val CRNN {val['accuracy']['crnn']:.2f}% "
            f"PSNR {val['psnr']:.2f}")
        if score > best_score + args.min_delta:
            best_score, best_state, best_epoch = score, copy.deepcopy(student.state_dict()), epoch
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if args.patience and epochs_without_gain >= args.patience:
                log(f'  early stop: no {args.select_by} gain above {args.min_delta:g} for {args.patience} epochs; '
                    f'keeping epoch {best_epoch}')
                break
    student.load_state_dict(best_state)
    tsrn_pruning.compact_gru_weights(student)
    return history, best_epoch


# ---------------------------------------------------------------- storage

def write_json(path, data):
    """Write through a temporary file, so an interrupted session never leaves a half-written file."""
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def start_or_resume_config(run_dir, args, parser, free_args, log=print):
    """Write config.json for a new run; for an existing run, refuse settings that would change results.

    Only the arguments in `free_args` may differ between sessions. A setting missing from an older
    config.json is compared with its default, which is the behaviour that run had before the
    setting existed.
    """
    path = os.path.join(run_dir, 'config.json')
    current = vars(args)
    if not os.path.exists(path):
        write_json(path, {'args': current, 'environment': environment_info(),
                          'started': time.strftime('%Y-%m-%d %H:%M:%S')})
        return
    saved = read_json(path)['args']
    changed = {k: (saved.get(k, parser.get_default(k)), v) for k, v in current.items()
               if k not in free_args and saved.get(k, parser.get_default(k)) != v}
    if changed:
        raise SystemExit(f'{run_dir} was started with different settings {changed}; '
                         'use the original settings or a new --run_name')
    log(f'resuming run in {run_dir}')


def save_atomic(obj, path):
    tmp = f'{path}.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


class RunLog:
    """Print and append to a log file in the run folder."""

    def __init__(self, path):
        self.path = path

    def __call__(self, message=''):
        print(message, flush=True)
        with open(self.path, 'a') as f:
            f.write(message + '\n')
