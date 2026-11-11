"""Self-check of TSRN pruning on random weights (CPU, under a minute). Run from src/:
    python check_gru_pruning.py

1. Dead units: units made exactly inactive are removed, so the output must not change.
   This checks that every group cuts the right rows and columns.
2. Taylor pruning of every group, a training step through the smaller GRUs, and a
   save/rebuild/load round trip.
3. Size and single-thread CPU time (one image) of conv-only, GRU-only and combined cuts.
"""
import statistics
import time

import torch

import tsrn_pruning
from model.tsrn import TSRN

TOLERANCE = 1e-4


def build():
    torch.manual_seed(0)  # same configuration as tsrn_experiment.build_tsrn
    return TSRN(scale_factor=2, width=128, height=32, STN=True, mask=True, srb_nums=5, hidden_units=32).eval()


def random_input(batch_size=2):
    return torch.rand(batch_size, 4, *tsrn_pruning.LR_SIZE, generator=torch.Generator().manual_seed(1))


@torch.no_grad()
def make_trunk_units_dead(model, units):
    """Trunk channels u and H+u become exactly 0 everywhere, and the GRU hidden units u stay at 0."""
    hidden = model.block2.gru1.gru.hidden_size
    channels = units + [hidden + unit for unit in units]
    head = model.block1[0]
    refine_conv, refine_bn = getattr(model, f'block{model.srb_nums + 2}')
    for layer in (head, refine_conv):
        layer.weight[channels] = 0
        layer.bias[channels] = 0
    refine_bn.bias[channels] = 0
    refine_bn.running_mean[channels] = 0
    for i in range(model.srb_nums):
        block = getattr(model, f'block{i + 2}')
        for gru in (block.gru1.gru, block.gru2.gru):
            for suffix in tsrn_pruning.DIRECTIONS:
                # Update gate z = 1, so h_t = h_(t-1) = h_0 = 0 for these units.
                getattr(gru, f'bias_ih_l0{suffix}')[[hidden + unit for unit in units]] = 1e4


@torch.no_grad()
def make_gru_inputs_dead(gru_block, features):
    gru_block.conv1.weight[features] = 0
    gru_block.conv1.bias[features] = 0


@torch.no_grad()
def make_conv1_channels_dead(block, channels):
    block.conv1.weight[channels] = 0
    block.conv1.bias[channels] = 0
    block.bn1.bias[channels] = 0
    block.bn1.running_mean[channels] = 0


def check_dead_units():
    model = build()
    trunk_units, gru_features, conv_channels = [0, 5, 17, 31], [2, 3, 40, 63], [1, 30, 62]
    make_trunk_units_dead(model, trunk_units)
    for i in range(model.srb_nums):
        block = getattr(model, f'block{i + 2}')
        make_gru_inputs_dead(block.gru1, gru_features)
        make_gru_inputs_dead(block.gru2, gru_features)
        make_conv1_channels_dead(block, conv_channels)
    x = random_input()
    with torch.no_grad():
        reference = model(x)
    for group in tsrn_pruning.prunable_groups(model, gru=True):
        if group.name == 'trunk':
            group.remove(trunk_units)
        elif group.name.endswith('.gru1.conv1') or group.name.endswith('.gru2.conv1'):
            group.remove(gru_features)
        elif group.name.endswith('.conv1'):
            group.remove(conv_channels)
    with torch.no_grad():
        difference = (model(x) - reference).abs().max().item()
    sizes = tsrn_pruning.channel_sizes(model, gru=True)
    print(f'1. dead units removed: trunk {sizes["trunk"]} units, GRU inputs {sizes["block2.gru1.conv1"]}, '
          f'conv1 {sizes["block2.conv1"]}; largest output change {difference:.1e}')
    assert sizes['trunk'] == 28 and sizes['block2.gru1.conv1'] == 60 and sizes['block2.conv1'] == 61, sizes
    assert difference < TOLERANCE, 'removing inactive units changed the output: wrong rows or columns were cut'


def check_taylor_round_trip():
    model = build().train()
    model(random_input(4)).pow(2).mean().backward()
    report = tsrn_pruning.prune(model, 0.5, 'taylor', round_to=8, gru_ratio=0.5)
    shape = tsrn_pruning.check_structure(model, 'cpu')
    model.train()
    loss = model(random_input(4)).pow(2).mean()
    loss.backward()  # gradients must flow through the rebuilt GRUs
    assert all(p.grad is not None for name, p in model.named_parameters() if 'gru' in name)
    sizes = tsrn_pruning.channel_sizes(model, gru=True)
    rebuilt = tsrn_pruning.shrink_to(build(), sizes)
    rebuilt.load_state_dict(model.state_dict())
    difference = tsrn_pruning.max_output_difference(model, rebuilt, 'cpu')
    print(f'2. Taylor pruning of {len(report)} groups: trunk {report["trunk"]}, '
          f'GRU input {report["block2.gru1.conv1"]}, conv {report["block2.conv1"]}; output {shape}; '
          f'save/load difference {difference:.1e}')
    assert report['trunk'] == (32, 16) and report['block2.gru1.conv1'] == (64, 32), report
    assert difference < TOLERANCE, 'a rebuilt model does not load to the same model'


@torch.no_grad()
def cpu_ms(model, runs=30):
    model.eval()
    x = random_input(1)
    for _ in range(3):
        model(x)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def compare_sizes():
    torch.set_num_threads(1)
    cuts = {'uncut': (0, 0), 'conv 50%': (0.5, 0), 'GRU 50%': (0, 0.5), 'conv 50% + GRU 50%': (0.5, 0.5),
            'conv 87.5%': (0.875, 0), 'conv 87.5% + GRU 50%': (0.875, 0.5)}
    print('3. size and CPU time (1 thread, 1 image, random weights; on Colab the timing is only a rough guide):')
    print(f'   {"cut":24s}{"params":>9s}{"MMACs":>8s}{"CPU ms":>8s}{"speed-up":>10s}')
    reference = None
    for label, (ratio, gru_ratio) in cuts.items():
        model = build()
        if ratio or gru_ratio:
            tsrn_pruning.prune(model, ratio, 'magnitude', round_to=8, gru_ratio=gru_ratio)
        ms = cpu_ms(model)
        reference = reference or ms
        print(f'   {label:24s}{tsrn_pruning.inference_parameters(model):>9,d}'
              f'{tsrn_pruning.inference_macs(model) / 1e6:>8.0f}{ms:>8.1f}{reference / ms:>9.2f}x')


if __name__ == '__main__':
    check_dead_units()
    check_taylor_round_trip()
    compare_sizes()
    print('All checks passed.')
