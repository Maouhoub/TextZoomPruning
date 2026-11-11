"""Structured pruning for TSRN: convolution channels and, on request, GRU units.

Torch-Pruning's automatic dependency tracing is not used on TSRN: it has no PixelShuffle
support, and it treats the GRU weights as unwrapped parameters whose last dimension it would
prune. The groups that can be pruned safely are listed explicitly instead. Convolutions are cut
with Torch-Pruning's low-level prune functions; a GRU is rebuilt smaller with the kept weights.

Convolution groups (ChannelGroup, kind 'conv'):
  - inside every sequential residual block: conv1 -> bn1 -> conv2, and conv2 -> bn2 -> gru1.conv1
  - the upsampler: conv -> PixelShuffle -> next conv, in units of r*r conv channels

GRU groups (kind 'gru', only with `gru=True`):
  - GruInputGroup, one per GruBlock: the output channels of its 1x1 conv, which are the input
    features of its GRU.
  - TrunkGroup, one for the whole model: the GRU hidden units. Every GRU output is added to the
    trunk (the 2H channels running from block1 to the upsampler), and GRU output channel d*H + u
    is hidden unit u of direction d. So one unit is hidden unit u, in both directions, of all the
    GRUs, together with trunk channels u and H + u in every layer that reads or writes the trunk.
    Removing a hidden unit removes its rows in all three gates and its recurrent column.

Always protected: the input of the first layer, the 4-channel output (RGB + mask), and the STN,
which only runs during training.
"""
import copy
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch_pruning as tp

from model.tsrn import GruBlock, RecurrentResidualBlock, UpsampleBLock

LR_SIZE = (16, 64)
SCALE = 2
TRAINING_ONLY_PREFIXES = ('stn_head', 'tps')
GRU_GATES = 3  # reset, update and new gate, stacked in this order in every GRU weight and bias
DIRECTIONS = ('', '_reverse')


# ---------------------------------------------------------------- importance

def element_scores(weight, criterion, group_name):
    """taylor: |w * dL/dw| (first-order Taylor estimate of the loss change); magnitude: w^2."""
    if criterion == 'taylor':
        if weight.grad is None:
            raise RuntimeError(f'{group_name}: Taylor importance needs gradients; run a backward pass first')
        return (weight * weight.grad).abs()
    if criterion == 'magnitude':
        return weight.pow(2)
    raise ValueError(f'unknown importance criterion: {criterion}')


def combine(layer_scores, criterion):
    """Score per unit: the mean over the group's layers of each layer's score for that unit
    (an L2 norm per layer for magnitude)."""
    if criterion == 'magnitude':
        layer_scores = [scores.sqrt() for scores in layer_scores]
    return torch.stack(layer_scores).mean(dim=0)


def conv_scores(conv, dim, criterion, group_name):
    """Score per output channel (dim 0) or per input channel (dim 1) of a convolution."""
    others = tuple(d for d in range(conv.weight.dim()) if d != dim)
    return element_scores(conv.weight, criterion, group_name).sum(dim=others)


# ---------------------------------------------------------------- GRUs

def gate_rows(units, hidden):
    """Rows of a GRU weight or bias that belong to these hidden units, in all three gates."""
    return [gate * hidden + unit for gate in range(GRU_GATES) for unit in units]


def hidden_unit_scores(gru, criterion, group_name):
    """Score per hidden unit, summed over both directions: its rows in the three gates (input and
    recurrent weights) plus its recurrent column (the state every gate reads at the next step)."""
    hidden = gru.hidden_size
    total = torch.zeros(hidden, device=gru.weight_hh_l0.device)
    for suffix in DIRECTIONS:
        for kind in ('ih', 'hh'):
            rows = element_scores(getattr(gru, f'weight_{kind}_l0{suffix}'), criterion, group_name).sum(dim=1)
            total += rows.view(GRU_GATES, hidden).sum(dim=0)
        total += element_scores(getattr(gru, f'weight_hh_l0{suffix}'), criterion, group_name).sum(dim=0)
    return total


def input_feature_scores(gru, criterion, group_name):
    """Score per input feature: its column in the input weights of both directions."""
    return sum(element_scores(getattr(gru, f'weight_ih_l0{suffix}'), criterion, group_name).sum(dim=0)
               for suffix in DIRECTIONS)


def copy_slice(source, target, rows, columns=None):
    """Copy the kept rows (and columns) of a parameter, and of its gradient if it has one."""
    def cut(tensor):
        tensor = tensor.index_select(0, rows)
        return tensor if columns is None else tensor.index_select(1, columns)

    with torch.no_grad():
        target.copy_(cut(source))
    if source.grad is not None:
        target.grad = cut(source.grad).clone()  # Taylor scores of later groups still need it
    target.requires_grad_(source.requires_grad)


def rebuild_gru(gru, keep_hidden=None, keep_input=None):
    """A smaller copy of a one-layer bidirectional GRU with only these hidden units and input features."""
    if gru.num_layers != 1 or not gru.bidirectional:
        raise ValueError('only one-layer bidirectional GRUs can be pruned')
    hidden = list(range(gru.hidden_size)) if keep_hidden is None else sorted(keep_hidden)
    inputs = list(range(gru.input_size)) if keep_input is None else sorted(keep_input)
    device, dtype = gru.weight_ih_l0.device, gru.weight_ih_l0.dtype
    smaller = nn.GRU(len(inputs), len(hidden), bias=gru.bias, batch_first=gru.batch_first, bidirectional=True,
                     device=device, dtype=dtype)
    rows = torch.tensor(gate_rows(hidden, gru.hidden_size), device=device)
    columns = {'ih': torch.tensor(inputs, device=device), 'hh': torch.tensor(hidden, device=device)}
    for suffix in DIRECTIONS:
        for kind in ('ih', 'hh'):
            name = f'weight_{kind}_l0{suffix}'
            copy_slice(getattr(gru, name), getattr(smaller, name), rows, columns[kind])
            if gru.bias:
                name = f'bias_{kind}_l0{suffix}'
                copy_slice(getattr(gru, name), getattr(smaller, name), rows)
    smaller.train(gru.training)
    smaller.flatten_parameters()
    return smaller


def keep_list(units, total):
    removed = set(units)
    return [i for i in range(total) if i not in removed]


# ---------------------------------------------------------------- groups

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
    kind = 'conv'

    @property
    def units(self):
        return self.consumer.in_channels

    def conv_channels(self, units):
        per_unit = self.shuffle ** 2
        return [unit * per_unit + i for unit in units for i in range(per_unit)]

    def importance(self, criterion):
        layer_scores = [conv_scores(self.conv, 0, criterion, self.name).view(self.units, -1).sum(dim=1)]
        if self.bn is not None:
            layer_scores.append(element_scores(self.bn.weight, criterion, self.name))
        layer_scores.append(conv_scores(self.consumer, 1, criterion, self.name))
        return combine(layer_scores, criterion)

    def remove(self, units):
        units = sorted(units)
        tp.prune_conv_out_channels(self.conv, self.conv_channels(units))
        if self.bn is not None:
            tp.prune_batchnorm_out_channels(self.bn, units)
        tp.prune_conv_in_channels(self.consumer, units)


@dataclass
class GruInputGroup:
    """Output channels of a GruBlock's 1x1 conv, which are the input features of its GRU."""
    name: str
    block: GruBlock
    kind = 'gru'

    @property
    def units(self):
        return self.block.conv1.out_channels

    def importance(self, criterion):
        return combine([conv_scores(self.block.conv1, 0, criterion, self.name),
                        input_feature_scores(self.block.gru, criterion, self.name)], criterion)

    def remove(self, units):
        units = sorted(units)
        keep = keep_list(units, self.units)
        tp.prune_conv_out_channels(self.block.conv1, units)
        self.block.gru = rebuild_gru(self.block.gru, keep_input=keep)


@dataclass
class TrunkGroup:
    """GRU hidden units, tied to the trunk channels (see the module docstring)."""
    name: str
    model: nn.Module
    kind = 'gru'

    def residual_blocks(self):
        return [getattr(self.model, f'block{i + 2}') for i in range(self.model.srb_nums)]

    def gru_blocks(self):
        return [gru for block in self.residual_blocks() for gru in (block.gru1, block.gru2)]

    def trunk_layers(self):
        """(convolution, dim) pairs holding trunk channels (dim 0 = output, dim 1 = input), and the BatchNorms."""
        head, activation = self.model.block1
        if activation.weight.numel() != 1:
            raise ValueError('block1 PReLU has one weight per channel; trunk pruning expects a single weight')
        refine_conv, refine_bn = getattr(self.model, f'block{self.model.srb_nums + 2}')
        upsampler = getattr(self.model, f'block{self.model.srb_nums + 3}')[0]
        tail_conv = upsampler.conv if isinstance(upsampler, UpsampleBLock) else upsampler
        convs = [(head, 0)]
        for block in self.residual_blocks():
            convs += [(block.conv1, 1), (block.gru2.conv1, 1)]
        convs += [(refine_conv, 1), (refine_conv, 0), (tail_conv, 1)]
        return convs, [refine_bn]

    @property
    def units(self):
        sizes = {block.gru.hidden_size for block in self.gru_blocks()}
        if len(sizes) != 1:
            raise RuntimeError(f'GRUs have different hidden sizes {sizes}; the trunk can not be indexed')
        return sizes.pop()

    def importance(self, criterion):
        hidden = self.units

        def per_unit(channel_scores):  # trunk channel d*H + u belongs to unit u
            return channel_scores.view(2, hidden).sum(dim=0)

        convs, bns = self.trunk_layers()
        layer_scores = [per_unit(conv_scores(conv, dim, criterion, self.name)) for conv, dim in convs]
        layer_scores += [per_unit(element_scores(bn.weight, criterion, self.name)) for bn in bns]
        layer_scores += [hidden_unit_scores(block.gru, criterion, self.name) for block in self.gru_blocks()]
        return combine(layer_scores, criterion)

    def remove(self, units):
        hidden = self.units
        units = sorted(units)
        channels = units + [hidden + unit for unit in units]
        convs, bns = self.trunk_layers()
        for conv, dim in convs:
            (tp.prune_conv_out_channels if dim == 0 else tp.prune_conv_in_channels)(conv, channels)
        for bn in bns:
            tp.prune_batchnorm_out_channels(bn, channels)
        keep = keep_list(units, hidden)
        for block in self.gru_blocks():
            block.gru = rebuild_gru(block.gru, keep_hidden=keep)


def prunable_groups(model, gru=False):
    """The convolution groups, then (with `gru`) the GRU input groups and the trunk group."""
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
    if gru:
        for name, module in model.named_children():
            if isinstance(module, RecurrentResidualBlock):
                groups.append(GruInputGroup(f'{name}.gru1.conv1', module.gru1))
                groups.append(GruInputGroup(f'{name}.gru2.conv1', module.gru2))
        groups.append(TrunkGroup('trunk', model))
    return groups


# ---------------------------------------------------------------- pruning

def units_to_remove(units, ratio, round_to):
    """Remove a multiple of `round_to` units, and always keep at least `round_to`."""
    count = int(units * ratio) // round_to * round_to
    return max(0, min(count, units - round_to))


def prune_to(model, targets, criterion):
    """Keep `targets[name]` units in every group named in `targets`, removing the least important first.

    Groups are pruned one after the other, in the order of prunable_groups; each is scored just
    before it is cut (the prune functions cut the gradients along with the weights).
    Returns {group name: (units before, units after)}.
    """
    report = {}
    for group in prunable_groups(model, gru=True):
        if group.name not in targets:
            continue
        before = group.units
        count = before - targets[group.name]
        if count < 0:
            raise ValueError(f'{group.name}: cannot grow from {before} to {targets[group.name]} units')
        if count > 0:
            scores = group.importance(criterion)
            group.remove(torch.argsort(scores)[:count].tolist())
        report[group.name] = (before, group.units)
    model.zero_grad(set_to_none=True)
    return report


def prune(model, ratio, criterion, round_to=4, gru_ratio=0.0):
    """Remove `ratio` of the units of every convolution group and `gru_ratio` of every GRU group,
    least important first."""
    targets = {}
    for group in prunable_groups(model, gru=gru_ratio > 0):
        group_ratio = gru_ratio if group.kind == 'gru' else ratio
        targets[group.name] = group.units - units_to_remove(group.units, group_ratio, round_to)
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


def channel_sizes(model, gru=False):
    """Units per group: the convolution groups, plus the GRU groups with `gru`."""
    return {group.name: group.units for group in prunable_groups(model, gru)}


def shrink_to(model, sizes):
    """Give an unpruned TSRN the sizes of a pruned one, so that its state dict loads.

    Groups missing from `sizes` keep their size (checkpoints saved before GRU pruning existed).
    """
    for group in prunable_groups(model, gru=True):
        if group.name in sizes and sizes[group.name] < group.units:
            group.remove(list(range(sizes[group.name], group.units)))
    return model


@torch.no_grad()
def max_output_difference(model_a, model_b, device, batch_size=2):
    """Largest absolute difference between the inference outputs of two models on the same random input."""
    modes = model_a.training, model_b.training
    model_a.eval(), model_b.eval()
    x = torch.rand(batch_size, 4, *LR_SIZE, device=device)
    difference = (model_a(x) - model_b(x)).abs().max().item()
    model_a.train(modes[0]), model_b.train(modes[1])
    return difference


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
