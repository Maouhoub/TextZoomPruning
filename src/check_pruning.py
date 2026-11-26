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
5. LEMMA with a small guidance network (lemma_fast.py): the shared forward pieces reproduce LEMMA,
   the copied SR branch gives LEMMA's output from LEMMA's guidance map, a training step moves only
   the guidance network, and a saved model loads back the same.
6. Checks of LEMMA's reader (check_lemma_reader.py): its pieces rebuild LEMMA, the timing prefixes end
   at LEMMA's guidance map, the part sizes add up to the guidance branch, and the true-word letters
   decode back to the words.
7. Slim LEMMA backbones (lemma_backbone.py): at full width with 3 layers the slim backbone loads LEMMA's
   backbone weights strictly and gives the same output; thinner and shallower ones keep the 512 x 8 x 32
   interface, and LEMMA with one still gives a 4 x 32 x 128 image.
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


def check_lemma_fast():
    import lemma_fast
    import lemma_teacher

    torch.manual_seed(0)
    lemma = lemma_teacher.LemmaSR().eval()
    x = random_input()
    with torch.no_grad():
        target_sr, target_map = lemma_teacher.outputs_with_guidance(lemma, x)
        same_as_lemma = (target_sr - lemma(x)).abs().max().item()
        # The student's SR branch is LEMMA's: fed LEMMA's own guidance map, it must give LEMMA's output.
        student = lemma_fast.LemmaFastSR(lemma, lemma_fast.GuidanceNet(64)).eval()
        features = student.block1(x)
        copied_branch = (lemma_teacher.sr_branch(student, features, target_map) - target_sr).abs().max().item()
        shapes = {width: tuple(lemma_fast.GuidanceNet(width)(features).shape) for width in (64, 128)}
    assert same_as_lemma < TOLERANCE and copied_branch < TOLERANCE
    assert all(shape == (2, 64, *tsrn_pruning.LR_SIZE) for shape in shapes.values())

    # One training step moves only GuidanceNet: the SR weights and batch-norm statistics stay as they were.
    before = {k: v.clone() for k, v in student.state_dict().items() if not k.startswith('guidance.')}
    lemma_fast.train_guidance_only(student)
    optimizer = torch.optim.Adam(student.guidance.parameters(), lr=1e-3)
    student(x).pow(2).mean().backward()
    optimizer.step()
    moved = [k for k, v in student.state_dict().items() if not k.startswith('guidance.') and not torch.equal(v, before[k])]
    with_grad = [n for n, p in student.named_parameters() if p.grad is not None and not n.startswith('guidance.')]
    assert not moved and not with_grad, f'the frozen SR branch changed: {moved or with_grad}'

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'lemma_fast.pth')
        lemma_fast.save(student, path)
        loaded = lemma_fast.load(path, 'cpu')
    difference = tsrn_pruning.max_output_difference(student, loaded, 'cpu')
    counts = lemma_fast.parameter_counts(student)
    print(f"5. LEMMA with a small guidance network: LEMMA's forward reproduced ({same_as_lemma:.1e}), its SR "
          f"branch copied ({copied_branch:.1e}), guidance map {shapes[64]}, a training step moves only the "
          f"guidance network, reload difference {difference:.1e}; {counts['sr_branch']:,} SR + "
          f"{counts['guidance_branch']:,} guidance parameters (width 64)")
    assert difference < TOLERANCE


@torch.no_grad()
def check_lemma_reader():
    import check_lemma_reader as reader_checks
    import lemma_teacher

    torch.manual_seed(0)
    lemma = lemma_teacher.LemmaSR().eval()
    x = random_input()
    rebuilt = (reader_checks.GuidedLemma(lemma, reader_checks.as_lemma)(x) - lemma(x)).abs().max().item()
    prefix = (reader_checks.GuidancePrefix(lemma, 'alignment')(x)
              - lemma_teacher.LemmaGuidance(lemma)(x)).abs().max().item()
    shapes = {part: tuple(reader_checks.GuidancePrefix(lemma, part)(x).shape) for part in reader_checks.PARTS}
    expected = {'ResNet45': (2, 512, 8, 32), 'transformer x3': (2, 512, 8, 32),
                'position attention + classifier': (2, 26, 37), 'language model x4': (2, 26, 37),
                'alignment': (2, 64, *tsrn_pruning.LR_SIZE)}
    params = reader_checks.part_parameters(lemma)
    charset = lemma.net.position_prior.language.charset
    words = ['hello', '42']
    truth = reader_checks.TrueWords(charset)(reader_checks.read(lemma.net.position_prior, reader_checks.upscale(x)), words)
    decoded = reader_checks.decode(truth['text_logits'], charset)
    vision = reader_checks.GuidedLemma(lemma, reader_checks.vision_only)(x)
    oracle = reader_checks.GuidedLemma(lemma, reader_checks.TrueWords(charset))(x, words)
    print(f"6. LEMMA's reader checks: LEMMA rebuilt from its pieces ({rebuilt:.1e}), last prefix = guidance map "
          f"({prefix:.1e}), part sizes {', '.join(f'{p} {n:,}' for p, n in params.items())}; true words decode to "
          f"{decoded}, lengths {truth['pt_lengths'].tolist()}")
    assert rebuilt < TOLERANCE and prefix < TOLERANCE
    assert shapes == expected, shapes
    assert sum(params.values()) == lemma_teacher.parameter_counts(lemma)['guidance_branch'], params
    assert decoded == words and truth['pt_lengths'].tolist() == [6, 3]
    assert tuple(vision.shape) == tuple(oracle.shape) == (2, 4, 32, 128)


@torch.no_grad()
def check_slim_backbone():
    import check_lemma_reader as reader_checks
    import lemma_backbone
    import lemma_teacher

    torch.manual_seed(0)
    lemma = lemma_teacher.LemmaSR().eval()
    original = lemma.net.position_prior.vision.backbone
    images = reader_checks.normalize(lemma.net.position_prior, reader_checks.upscale(random_input()))
    full = lemma_backbone.SlimBackbone(1.0, 3).eval()
    full.load_state_dict(original.state_dict())  # strict: same names and shapes as LEMMA's backbone
    same = (full(images) - original(images)).abs().max().item()
    sizes = {}
    for layers, width in ((2, 0.75), (1, 0.5), (0, 0.35)):
        slim = lemma_backbone.SlimBackbone(width, layers).eval()
        assert tuple(slim(images).shape) == (2, 512, 8, 32), (layers, width)
        sizes[f'L{layers}_w{width:g}'] = sum(p.numel() for p in slim.parameters())
    output = lemma_backbone.with_backbone(lemma, lemma_backbone.SlimBackbone(0.5, 1))(random_input())
    print(f"7. slim LEMMA backbones: full width loads LEMMA's backbone, same output ({same:.1e}); "
          f"{sum(p.numel() for p in original.parameters()):,} parameters vs "
          + ', '.join(f'{name} {n:,}' for name, n in sizes.items()) + f'; LEMMA with L1_w0.5 gives {tuple(output.shape)}')
    assert same < TOLERANCE and tuple(output.shape) == (2, 4, 32, 128)
    assert lemma.net.position_prior.vision.backbone is original  # with_backbone left LEMMA unchanged


if __name__ == '__main__':
    for case, parts in DEPTH_CASES.items():
        check_dead_units(case, parts)
    for case, parts in DEPTH_CASES.items():
        check_taylor_round_trip(case, parts)
    check_checkpoint_loading()
    check_lemma()
    check_lemma_fast()
    check_lemma_reader()
    check_slim_backbone()
    print('All checks passed.')
