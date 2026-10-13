"""Profile TSRN inference: how much time goes to GRUs, convolutions and the rest, on GPU and CPU.

Run from src/:  python profile_tsrn.py --resume <checkpoint>
"""
import argparse
import os
import statistics
import time
from collections import defaultdict

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from model.tsrn import TSRN

LR_HEIGHT, LR_WIDTH = 16, 64
REPORTED_CATEGORIES = ('GRU', 'conv', 'batchnorm', 'other')
# First matching keyword wins; matched against lower-cased top-level aten op names.
CATEGORY_KEYWORDS = (
    ('GRU', ('gru', 'rnn')),
    ('conv', ('conv',)),
    ('batchnorm', ('batch_norm',)),
)


def categorize(op_name):
    name = op_name.lower()
    for category, keywords in CATEGORY_KEYWORDS:
        if any(keyword in name for keyword in keywords):
            return category
    return 'other'


def build_model(checkpoint, device):
    # Same configuration as training with --STN --mask; the STN is skipped in eval mode.
    model = TSRN(scale_factor=2, width=128, height=32, STN=True, mask=True, srb_nums=5, hidden_units=32)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict_G'])
    return model.to(device).eval()


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def median_latency_ms(model, x, warmup, repeats):
    for _ in range(warmup):
        model(x)
    synchronize(x.device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(x)
        synchronize(x.device)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def op_time_us(event, device):
    if device.type == 'cuda':
        # Renamed from cuda_time_total in recent PyTorch releases.
        return event.device_time_total if hasattr(event, 'device_time_total') else event.cuda_time_total
    return event.cpu_time_total


@torch.no_grad()
def time_per_op_us(model, x, repeats):
    """Time per top-level op, summed over `repeats` forward passes.

    Only top-level ops are counted: their totals include the ops and kernels they call,
    so nothing is counted twice. On GPU the time is kernel time; on CPU it is CPU time.
    """
    activities = [ProfilerActivity.CPU]
    if x.device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        for _ in range(repeats):
            model(x)
        synchronize(x.device)

    per_op = defaultdict(float)
    for event in prof.events():
        if event.device_type == DeviceType.CPU and event.cpu_parent is None:
            per_op[event.name] += op_time_us(event, x.device)
    return per_op


def report(label, batch_size, latency_ms, per_op, profile_repeats, device):
    total = sum(per_op.values())
    per_category = defaultdict(float)
    for name, us in per_op.items():
        per_category[categorize(name)] += us

    shares = '  '.join(f'{c} {per_category[c] / total * 100:4.1f}%' for c in REPORTED_CATEGORIES)
    print(f'{label:18s} batch {batch_size:3d}: {latency_ms:9.2f} ms per batch | {shares}')
    if device.type == 'cuda':
        # Low values mean the GPU mostly waits for kernel launches, so fewer channels barely help.
        kernel_ms = total / profile_repeats / 1000
        print(f'{"":18s} GPU busy {kernel_ms / latency_ms * 100:4.0f}% of the measured latency')
    top_ops = sorted(per_op.items(), key=lambda item: -item[1])[:6]
    print(f'{"":18s} top ops: ' + ', '.join(f'{name} {us / total * 100:.1f}%' for name, us in top_ops))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', default='', help='TSRN checkpoint; weights do not change timing')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 64])
    parser.add_argument('--cpu_threads', type=int, nargs='+', default=[1, os.cpu_count()])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--cpu_warmup', type=int, default=3)
    parser.add_argument('--cpu_repeats', type=int, default=10)
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
    print(f'torch {torch.__version__} | GPU: {gpu_name} | CPU cores: {os.cpu_count()} | '
          f'input 4x{LR_HEIGHT}x{LR_WIDTH} (RGB + mask)')

    settings = []
    if torch.cuda.is_available():
        settings.append((torch.device('cuda'), None, args.warmup, args.repeats))
    for threads in dict.fromkeys(args.cpu_threads):
        settings.append((torch.device('cpu'), threads, args.cpu_warmup, args.cpu_repeats))

    for device, threads, warmup, repeats in settings:
        if threads:
            torch.set_num_threads(threads)
        label = 'GPU' if device.type == 'cuda' else f'CPU {threads} thread(s)'
        model = build_model(args.resume, device)
        for batch_size in args.batch_sizes:
            x = torch.rand(batch_size, 4, LR_HEIGHT, LR_WIDTH, device=device)
            latency_ms = median_latency_ms(model, x, warmup, repeats)
            profile_repeats = min(10, repeats)
            per_op = time_per_op_us(model, x, profile_repeats)
            report(label, batch_size, latency_ms, per_op, profile_repeats, device)


if __name__ == '__main__':
    main()
