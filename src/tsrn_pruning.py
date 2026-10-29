"""Structured channel pruning for TSRN.

Torch-Pruning's automatic dependency tracing is not used on TSRN: it has no PixelShuffle
support, and it treats the GRU weights as unwrapped parameters whose last dimension it would
prune. The channel groups that can be pruned safely are listed explicitly instead, and only
Torch-Pruning's low-level prune functions are used to remove them.

Prunable (one ChannelGroup each):
  - inside every sequential residual block: conv1 -> bn1 -> conv2, and conv2 -> bn2 -> gru1.conv1
  - the upsampler: conv -> PixelShuffle -> next conv, in units of r*r conv channels

Protected:
  - the trunk channels (block1, block7 and every residual addition depend on them)
  - every GruBlock (its 1x1 conv output and the bidirectional GRU): GRU pruning is not supported
  - the input layer and the 4-channel output (RGB + mask)
  - the STN, which only runs during training
"""
import copy
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch_pruning as tp

from model.tsrn import RecurrentResidualBlock, UpsampleBLock

LR_SIZE = (16, 64)
SCALE = 2
TRAINING_ONLY_PREFIXES = ('stn_head', 'tps')


@dataclass
class ChannelGroup:
    """Output channels of `conv` (and its BatchNorm) that feed the input channels of `consumer`.

    With a PixelShuffle of factor r in between, each consumer input channel is fed by r*r
    consecutive conv output channels. One prunable unit is one consumer input channel.
    """
    name: str
    conv: nn.Conv2d
    consumer: nn.Conv2d
    bn: Optional[nn.BatchNorm2d] = None
    shuffle: int = 1

    @property
    def units(self):
        return self.consumer.in_channels

    def conv_channels(self, units):
        per_unit = self.shuffle ** 2
        return [unit * per_unit + i for unit in units for i in range(per_unit)]


def prunable_groups(model):
    groups = []
    for name, module in model.named_children():
        if isinstance(module, RecurrentResidualBlock):
            groups.append(ChannelGroup(f'{name}.conv1', module.conv1, module.conv2, bn=module.bn1))
            groups.append(ChannelGroup(f'{name}.conv2', module.conv2, module.gru1.conv1, bn=module.bn2))

    tail_name = f'block{model.srb_nums + 3}'
    tail = list(getattr(model, tail_name))
    for i, layer in enumerate(tail[:-1]):
        if isinstance(layer, UpsampleBLock):
            following = tail[i + 1]
            consumer = following.conv if isinstance(following, UpsampleBLock) else following
            groups.append(ChannelGroup(f'{tail_name}.{i}.conv', layer.conv, consumer,
                                       shuffle=layer.pixel_shuffle.upscale_factor))
    return groups


def unit_importance(group, criterion):
    """Score per unit: the mean over the group's layers of each layer's score for that unit.

    taylor:    sum of |w * dL/dw| (first-order Taylor estimate of the loss change)
    magnitude: L2 norm of the weights
    """
    def per_element(weight):
        if criterion == 'taylor':
            if weight.grad is None:
                raise RuntimeError(f'{group.name}: Taylor importance needs gradients; run a backward pass first')
            return (weight * weight.grad).abs()
        if criterion == 'magnitude':
            return weight.pow(2)
        raise ValueError(f'unknown importance criterion: {criterion}')

    conv_scores = per_element(group.conv.weight).sum(dim=(1, 2, 3)).view(group.units, -1).sum(dim=1)
    layer_scores = [conv_scores]
    if group.bn is not None:
        layer_scores.append(per_element(group.bn.weight))
    layer_scores.append(per_element(group.consumer.weight).sum(dim=(0, 2, 3)))
    if criterion == 'magnitude':
        layer_scores = [scores.sqrt() for scores in layer_scores]
    return torch.stack(layer_scores).mean(dim=0)


def units_to_remove(units, ratio, round_to):
    """Remove a multiple of `round_to` units, and always keep at least `round_to`."""
    count = int(units * ratio) // round_to * round_to
    return max(0, min(count, units - round_to))


def remove_units(group, units):
    units = sorted(units)
    tp.prune_conv_out_channels(group.conv, group.conv_channels(units))
    if group.bn is not None:
        tp.prune_batchnorm_out_channels(group.bn, units)
    tp.prune_conv_in_channels(group.consumer, units)


def prune_to(model, targets, criterion):
    """Keep `targets[name]` units in every prunable group, removing the least important first.

    Returns {group name: (units before, units after)}.
    """
    report = {}
    for group in prunable_groups(model):
        before = group.units
        count = before - targets[group.name]
        if count < 0:
            raise ValueError(f'{group.name}: cannot grow from {before} to {targets[group.name]} units')
        if count > 0:
            scores = unit_importance(group, criterion)
            remove_units(group, torch.argsort(scores)[:count].tolist())
        report[group.name] = (before, group.units)
    model.zero_grad(set_to_none=True)
    return report


def prune(model, ratio, criterion, round_to=4):
    """Remove `ratio` of the units of every prunable group (same ratio for all), least important first."""
    targets = {g.name: g.units - units_to_remove(g.units, ratio, round_to) for g in prunable_groups(model)}
    return prune_to(model, targets, criterion)


def keep_schedule(units, target_ratio, steps, schedule, round_to):
    """Units to keep after each of `steps` pruning steps, ending at `target_ratio` removed.

    linear: the same amount removed at every step. cubic: large steps first, as in Chapter 1
    (Torch-Pruning's CubicScheduler). Counts are multiples of `round_to`, never below it.
    """
    kept = []
    for step in range(1, steps + 1):
        progress = step / steps
        if schedule == 'linear':
            ratio = target_ratio * progress
        elif schedule == 'cubic':
            ratio = target_ratio * (1 - (1 - progress) ** 3)
        else:
            raise ValueError(f'unknown schedule: {schedule}')
        kept.append(max(round_to, units - round(units * ratio / round_to) * round_to))
    if any(later >= earlier for earlier, later in zip([units] + kept, kept)):
        raise ValueError(f'schedule {kept} has steps that remove nothing; use fewer steps or a smaller round_to')
    return kept


def compact_gru_weights(model):
    """Re-pack GRU weights into one memory block for cuDNN.

    Copying a model (deepcopy, .to()) splits them, and cuDNN then re-packs them on every call,
    which slows the GRUs and distorts latency measurements.
    """
    for module in model.modules():
        if isinstance(module, nn.RNNBase):
            module.flatten_parameters()
    return model


def channel_sizes(model):
    return {group.name: group.units for group in prunable_groups(model)}


def shrink_to(model, sizes):
    """Give an unpruned TSRN the channel sizes of a pruned one, so that its state dict loads."""
    for group in prunable_groups(model):
        remove_units(group, list(range(sizes[group.name], group.units)))
    return model


def accumulate_gradients(model, loader, loss_fn, batches, device):
    """Gradients of `loss_fn(sr, hr, labels)` over `batches` batches, for Taylor importance."""
    model.train()
    model.zero_grad(set_to_none=True)
    for i, (images_hr, images_lr, labels) in enumerate(loader):
        if i == batches:
            break
        loss = loss_fn(model(images_lr.to(device)), images_hr.to(device), labels) / batches
        loss.backward()


@torch.no_grad()
def check_structure(model, device, batch_size=2):
    """One inference pass on random input: the output must be a finite 4 x 32 x 128 image."""
    was_training = model.training
    model.eval()
    output = model(torch.rand(batch_size, 4, *LR_SIZE, device=device))
    model.train(was_training)
    expected = (batch_size, 4, LR_SIZE[0] * SCALE, LR_SIZE[1] * SCALE)
    if tuple(output.shape) != expected:
        raise RuntimeError(f'pruned TSRN outputs {tuple(output.shape)}, expected {expected}')
    if not torch.isfinite(output).all():
        raise RuntimeError('pruned TSRN outputs non-finite values')
    return tuple(output.shape)


def inference_parameters(model):
    """Parameters used at inference: everything except the training-only STN."""
    return sum(p.numel() for name, p in model.named_parameters() if not name.startswith(TRAINING_ONLY_PREFIXES))


def inference_macs(model):
    """Multiply-accumulates for one LR image, counted by ptflops (module hooks, including the GRUs)."""
    from ptflops import get_model_complexity_info
    counted = copy.deepcopy(model).cpu().eval()  # ptflops attaches hooks and counters to the model
    macs, _ = get_model_complexity_info(counted, (4, *LR_SIZE), as_strings=False, backend='pytorch',
                                        print_per_layer_stat=False, verbose=False)
    return int(macs)


def split_indices(n_total, val_fraction, seed):
    """Deterministic train/validation split of the training set indices."""
    n_val = round(n_total * val_fraction)
    val = sorted(random.Random(seed).sample(range(n_total), n_val))
    val_set = set(val)
    return [i for i in range(n_total) if i not in val_set], val
