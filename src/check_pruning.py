"""Self-check of TSRN pruning on random weights (CPU, a few seconds). Run from src/:
    python check_pruning.py

Each check runs on the full model and on a depth-pruned one (a block and two GRUs removed):
1. Dead units: units made exactly inactive are removed, so the output must not change.
   This checks that every group cuts the right rows and columns.
2. Taylor pruning of every group, a training step through the smaller GRUs, and a
   save/rebuild/load round trip.
3. Checkpoints saved like train_tsrn_baseline.py (full, or shallow with --remove_parts) load back
   with tsrn_experiment.load_tsrn to the same model.
4. LEMMA's network (copied code, random weights) runs, its guidance branch runs alone, and a file in
   the layout of LEMMA's release loads back strictly to the same model.
"""
import os
import tempfile

import torch

import tsrn_depth
import tsrn_pruning
from model.tsrn import TSRN, GruBlock, RecurrentResidualBlock

TOLERANCE = 1e-4
DEPTH_CASES = {'full model': [], 'depth-pruned': ['block6', 'block2.gru1', 'block3.gru2']}


def build(parts=()):
    torch.manual_seed(0)  # same configuration as tsrn_experiment.build_tsrn
    model = TSRN(scale_factor=2, width=128, height=32, STN=True, mask=True, srb_nums=5, hidden_units=32)
    return tsrn_depth.remove_parts(model, parts).eval()


def random_input(batch_size=2):
    return torch.rand(batch_size, 4, *tsrn_pruning.LR_SIZE, generator=torch.Generator().manual_seed(1))


def residual_blocks(model):
    blocks = (getattr(model, name) for name in tsrn_depth.residual_block_names(model))
    return [block for block in blocks if isinstance(block, RecurrentResidualBlock)]


def gru_blocks(block):
    return [gru for gru in (block.gru1, block.gru2) if isinstance(gru, GruBlock)]


@torch.no_grad()
def make_dead(model, trunk_units, gru_features, conv_channels):
    """Make these units exactly inactive: their outputs are 0 wherever they are read."""
    hidden = model.block1[0].out_channels // 2
    trunk = trunk_units + [hidden + unit for unit in trunk_units]
    head = model.block1[0]
    refine_conv, refine_bn = getattr(model, f'block{model.srb_nums + 2}')
    for layer in (head, refine_conv):
        layer.weight[trunk] = 0
        layer.bias[trunk] = 0
    refine_bn.bias[trunk] = 0
    refine_bn.running_mean[trunk] = 0
    for block in residual_blocks(model):
        if not isinstance(block.gru1, GruBlock):  # conv2 -> bn2 writes the trunk
            block.conv2.weight[trunk] = 0
            block.conv2.bias[trunk] = 0
            block.bn2.bias[trunk] = 0
            block.bn2.running_mean[trunk] = 0
        for gru_block in gru_blocks(block):
            for suffix in tsrn_pruning.DIRECTIONS:
                # Update gate z = 1, so h_t = h_(t-1) = h_0 = 0 for the trunk units.
                getattr(gru_block.gru, f'bias_ih_l0{suffix}')[[hidden + unit for unit in trunk_units]] = 1e4
            gru_block.conv1.weight[gru_features] = 0
            gru_block.conv1.bias[gru_features] = 0
        block.conv1.weight[conv_channels] = 0
        block.conv1.bias[conv_channels] = 0
        block.bn1.bias[conv_channels] = 0
        block.bn1.running_mean[conv_channels] = 0


def check_dead_units(case, parts):
    model = build(parts)
    trunk_units, gru_features, conv_channels = [0, 5, 17, 31], [2, 3, 40, 63], [1, 30, 62]
    make_dead(model, trunk_units, gru_features, conv_channels)
    x = random_input()
    with torch.no_grad():
        reference = model(x)
    for group in tsrn_pruning.prunable_groups(model, gru=True):
        if isinstance(group, tsrn_pruning.TrunkGroup):
            group.remove(trunk_units)
        elif isinstance(group, tsrn_pruning.GruInputGroup):
            group.remove(gru_features)
        elif group.name.endswith('.conv1'):
            group.remove(conv_channels)
    with torch.no_grad():
        difference = (model(x) - reference).abs().max().item()
    sizes = tsrn_pruning.channel_sizes(model, gru=True)
    print(f'1. {case}: dead units removed (trunk {sizes["trunk"]} units, GRU inputs {sizes["block4.gru1.conv1"]}, '
          f'conv1 {sizes["block4.conv1"]}); largest output change {difference:.1e}')
    assert sizes['trunk'] == 28 and sizes['block4.gru1.conv1'] == 60 and sizes['block4.conv1'] == 61, sizes
    for part in parts:  # no group may be left for a removed block or GRU
        name, _, gru = part.partition('.')
        assert not any(key.startswith(f'{part}.' if gru else f'{name}.') for key in sizes), (part, sizes)
        if gru == 'gru1':  # conv2 now writes the trunk, so it has no group of its own
            assert f'{name}.conv2' not in sizes, (part, sizes)
    assert difference < TOLERANCE, 'removing inactive units changed the output: wrong rows or columns were cut'


def check_taylor_round_trip(case, parts):
    model = build(parts).train()
    model(random_input(4)).pow(2).mean().backward()
    report = tsrn_pruning.prune(model, 0.5, 'taylor', round_to=8, gru_ratio=0.5)
    shape = tsrn_pruning.check_structure(model, 'cpu')
    model.train()
    model(random_input(4)).pow(2).mean().backward()  # gradients must flow through the rebuilt GRUs
    assert all(p.grad is not None for name, p in model.named_parameters() if 'gru' in name)
    sizes = tsrn_pruning.channel_sizes(model, gru=True)
    rebuilt = tsrn_pruning.shrink_to(build(tsrn_depth.removed_parts(model)), sizes)
    rebuilt.load_state_dict(model.state_dict())
    difference = tsrn_pruning.max_output_difference(model, rebuilt, 'cpu')
    print(f'2. {case}: Taylor pruning of {len(report)} groups (trunk {report["trunk"]}, '
          f'GRU input {report["block4.gru1.conv1"]}, conv {report["block4.conv1"]}); output {shape}; '
          f'{tsrn_depth.gru_steps(model)} GRU steps; save/load difference {difference:.1e}')
    assert report['trunk'] == (32, 16) and report['block4.gru1.conv1'] == (64, 32), report
    assert difference < TOLERANCE, 'a rebuilt model does not load to the same model'


def check_checkpoint_loading():
    import tsrn_experiment as exp  # needs the full project environment (config, recognizers)

    for case, parts in {'full TSRN': [], 'shallow TSRN (2 blocks)': ['block2', 'block3', 'block4']}.items():
        model = build(parts)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'final.pth')
            torch.save({'state_dict_G': model.state_dict(), 'channels': tsrn_pruning.channel_sizes(model, gru=True),
                        'depth': tsrn_depth.removed_parts(model), 'info': {'name': case}}, path)
            loaded = exp.load_tsrn(path, 'cpu')
        difference = tsrn_pruning.max_output_difference(model, loaded, 'cpu')
        print(f'3. {case}: saved like train_tsrn_baseline.py, load_tsrn difference {difference:.1e}, '
              f'removed {tsrn_depth.removed_parts(loaded)}')
        assert tsrn_depth.removed_parts(loaded) == parts and difference < TOLERANCE


@torch.no_grad()
def check_lemma():
    import lemma_teacher

    torch.manual_seed(0)
    lemma = lemma_teacher.LemmaSR().eval()
    x = random_input()
    output = lemma(x)
    guidance = lemma_teacher.LemmaGuidance(lemma)(x)
    assert output.shape == (2, 4, 32, 128) and guidance.shape == (2, 64, *tsrn_pruning.LR_SIZE)
    assert (lemma.reconstruct(lemma.align(x)) - output).abs().max() < TOLERANCE
    with tempfile.TemporaryDirectory() as folder:  # the released file's layout, saved from a multi-GPU model
        path = os.path.join(folder, 'lemma.pth')
        torch.save({'state_dict_G': {f'module.{k}': v for k, v in lemma.net.state_dict().items()}}, path)
        loaded = lemma_teacher.load_lemma(path, 'cpu')
    difference = (loaded(x) - output).abs().max().item()
    counts = lemma_teacher.parameter_counts(lemma)
    print(f"4. LEMMA network (random weights): output {tuple(output.shape)}, guidance {tuple(guidance.shape)}, "
          f"reload difference {difference:.1e}, {counts['inference']:,} inference parameters "
          f"({counts['guidance_branch']:,} in the guidance branch)")
    assert difference < TOLERANCE and not any(p.requires_grad for p in loaded.parameters())


if __name__ == '__main__':
    for case, parts in DEPTH_CASES.items():
        check_dead_units(case, parts)
    for case, parts in DEPTH_CASES.items():
        check_taylor_round_trip(case, parts)
    check_checkpoint_loading()
    check_lemma()
    print('All checks passed.')
