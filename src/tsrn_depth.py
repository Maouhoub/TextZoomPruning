"""Depth pruning for TSRN: remove whole sequential residual blocks (SRBs) or whole GRUs.

TSRN's single-image latency is set by how many layers and GRU time steps run one after another,
not by how wide they are, so cutting width barely helps. Here depth is cut instead. In each SRB
(block2 ... block6) one can remove:

  - 'blockN': the whole block; the trunk passes through unchanged;
  - 'blockN.gru1': its vertical GRU (with the 1x1 conv before it), which scans the 16 rows; the
    residual branch then ends at bn2, and conv2 writes the trunk directly;
  - 'blockN.gru2': its horizontal GRU (with the 1x1 conv before it), which scans the 64 columns;
    the block then returns x + residual, a plain residual block.

Removed modules become nn.Identity and everything else keeps its weights. removed_parts() lists
the removals of a model, and remove_parts() replays them on a fresh model before its weights load.
"""
import torch.nn as nn

from model.tsrn import GruBlock, RecurrentResidualBlock
from tsrn_pruning import LR_SIZE

GRUS = ('gru1', 'gru2')
STEPS_PER_DIRECTION = {'gru1': LR_SIZE[0], 'gru2': LR_SIZE[1]}  # 16 rows, 64 columns


def residual_block_names(model):
    return [f'block{i + 2}' for i in range(model.srb_nums)]


def remove_parts(model, parts):
    """Remove each part, e.g. ['block6', 'block2.gru1']: a whole SRB, or one of its GRUs."""
    trunk = model.block1[0].out_channels
    for part in parts:
        name, _, gru = part.partition('.')
        if name not in residual_block_names(model):
            raise ValueError(f'{part}: {name} is not a sequential residual block')
        block = getattr(model, name)
        if not isinstance(block, RecurrentResidualBlock):
            raise ValueError(f'{part}: {name} is already removed')
        if not gru:
            setattr(model, name, nn.Identity())
        elif gru in GRUS and isinstance(getattr(block, gru), GruBlock):
            if gru == 'gru1' and block.conv2.out_channels != trunk:
                raise ValueError(f'{part}: conv2 has {block.conv2.out_channels} channels but must write the '
                                 f'{trunk}-channel trunk; remove GRUs before pruning channels')
            setattr(block, gru, nn.Identity())
        else:
            raise ValueError(f'cannot remove {part}')
    return model


def remaining_blocks(model):
    return [name for name in residual_block_names(model) if isinstance(getattr(model, name), RecurrentResidualBlock)]


def choose_blocks(model, count, score, log=print):
    """Remove `count` SRBs one at a time (greedy): each time try removing every remaining block and
    remove the one that leaves `score(model)` highest (e.g. validation reading accuracy).

    Returns one record per removal: the score with each candidate removed, and the block chosen.
    """
    history = []
    for step in range(1, count + 1):
        scores = {}
        for name in remaining_blocks(model):
            block = getattr(model, name)
            setattr(model, name, nn.Identity())
            scores[name] = score(model)
            setattr(model, name, block)  # put it back before trying the next one
        chosen = max(scores, key=scores.get)
        remove_parts(model, [chosen])
        history.append({'step': step, 'scores_without': scores, 'removed': chosen})
        log(f'     removal {step}: score without each block '
            + ', '.join(f'{name} {value:.2f}' for name, value in scores.items()) + f' -> remove {chosen}')
    return history


def removed_parts(model):
    parts = []
    for name in residual_block_names(model):
        block = getattr(model, name)
        if isinstance(block, RecurrentResidualBlock):
            parts += [f'{name}.{gru}' for gru in GRUS if not isinstance(getattr(block, gru), GruBlock)]
        else:
            parts.append(name)
    return parts


def gru_steps(model):
    """GRU time steps per image, both directions counted (on a CPU they run one after another)."""
    steps = 0
    for name in residual_block_names(model):
        block = getattr(model, name)
        if isinstance(block, RecurrentResidualBlock):
            steps += sum(2 * STEPS_PER_DIRECTION[gru] for gru in GRUS if isinstance(getattr(block, gru), GruBlock))
    return steps
