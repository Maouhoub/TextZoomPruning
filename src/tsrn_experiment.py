"""Shared pieces of the TSRN pruning experiments: data, evaluation, fine-tuning, latency, storage."""
import copy
import json
import os
import platform
import random
import statistics
import subprocess
from importlib import metadata
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


# ---------------------------------------------------------------- data

def make_loader(data, batch_size, cfg, shuffle):
    collate = alignCollate_real(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale, mask=True)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=int(cfg.workers),
                      collate_fn=collate, drop_last=shuffle)


def load_data(cfg, args, split_path, with_test):
    """Train/validation split of the TextZoom training set, and the three test splits.

    The split is written to `split_path` the first time and checked against it afterwards, so a
    resumed run can never silently use a different validation set.
    """
    train_full = ConcatDataset([lmdbDataset_real(root=d, voc_type=cfg.voc_type, max_len=cfg.max_len)
                                for d in cfg.train_data_dir])
    train_idx, val_idx = tsrn_pruning.split_indices(len(train_full), args.val_fraction, args.seed)
    split = {'train_data': [os.path.basename(os.path.normpath(d)) for d in cfg.train_data_dir],
             'n_total': len(train_full), 'val_fraction': args.val_fraction, 'seed': args.seed,
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
                           test=test, n_train=len(train_idx), n_val=len(val_idx))


# ---------------------------------------------------------------- measurements

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


# ---------------------------------------------------------------- fine-tuning

def fine_tune(student, teacher, train_loader, val_loader, textbase, val_recognizers, args, device, log=print):
    """Train with the original TSRN loss plus output distillation; keep the best epoch on validation.

    Runs at most `args.epochs` epochs. With `args.patience` > 0 it stops early once the validation
    metric (`args.select_by`) has not improved by more than `args.min_delta` for `patience` epochs,
    as in Chapter 1. Teacher and student see the same STN-aligned input, so the distillation target
    matches. Returns (history, best epoch).
    """
    criterion = ImageLoss(gradient=True, loss_weight=[1, 1e-4]).to(device)
    optimizer = optim.Adam(student.parameters(), lr=args.lr, betas=(0.5, 0.999))
    best_score, best_state, best_epoch, history = -float('inf'), None, None, []
    epochs_without_gain = 0
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
