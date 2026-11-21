"""LEMMA with its text-guidance branch replaced by a small network (notebook section 21).

LEMMA's SR branch (1.4M parameters) is steered by a 64 x 16 x 64 guidance map. LEMMA makes that map
with its text-guidance branch (an ABINet-style recognizer and alignment transformers: 36.5M of its
37.9M parameters, 54-74% of its time). Here GuidanceNet, a small dilated CNN, makes the map from the
same first-layer features instead, while the SR branch keeps LEMMA's released weights.
"""
import copy

import torch
import torch.nn as nn

import lemma_teacher
import tsrn_experiment as exp

# (height, width) dilation of each residual block: together the blocks see the whole 16 x 64 image.
DILATIONS = ((1, 1), (1, 2), (1, 4), (1, 8), (2, 16))


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.body(x))


class GuidanceNet(nn.Module):
    """Small stand-in for LEMMA's guidance branch: first-layer features (64 x 16 x 64) -> guidance map
    (64 x 16 x 64). Convolutions only, so nothing runs step by step."""

    def __init__(self, width=64, channels=64):
        super().__init__()
        self.width = width
        self.stem = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResidualBlock(width, dilation) for dilation in DILATIONS])
        self.head = nn.Conv2d(width, channels, 1)

    def forward(self, features):
        return self.head(self.blocks(self.stem(features)))


class LemmaFastSR(nn.Module):
    """LEMMA's SR branch (copied from a LEMMA model) steered by a GuidanceNet, with our TSRN's interface."""

    def __init__(self, lemma, guidance):
        super().__init__()
        net = lemma.net
        self.srb_nums, self.block_range = net.srb_nums, list(net.block_range)
        for i in range(1, net.srb_nums + 4):  # block1 (first layer) ... block8 (upsampler)
            setattr(self, f'block{i}', copy.deepcopy(getattr(net, f'block{i}')))
        self.guidance = guidance

    def align(self, x):
        return x

    def reconstruct(self, x):
        features = self.block1(x)
        return lemma_teacher.sr_branch(self, features, self.guidance(features))

    def forward(self, x):
        return self.reconstruct(x)


def train_guidance_only(model):
    """Train mode for GuidanceNet only. The SR branch stays frozen and in eval mode, so its batch-norm
    statistics do not move. Its GRUs are switched to train mode only because cuDNN cannot run a GRU
    backward pass in eval mode; they have no dropout, so their output is unchanged."""
    model.eval()
    model.guidance.train()
    for module in model.modules():
        if isinstance(module, nn.RNNBase):
            assert module.dropout == 0, 'a GRU with dropout would change its output in train mode'
            module.train()
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith('guidance.')


def parameter_counts(model):
    guidance = sum(p.numel() for p in model.guidance.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {'inference': total, 'guidance_branch': guidance, 'sr_branch': total - guidance}


def save(model, path, **info):
    exp.save_atomic({'state_dict': model.state_dict(), 'width': model.guidance.width, 'info': info}, path)


def load(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = LemmaFastSR(lemma_teacher.LemmaSR(), GuidanceNet(saved['width']))
    model.load_state_dict(saved['state_dict'])
    return model.to(device).eval()


def saved_info(path):
    return torch.load(path, map_location='cpu', weights_only=True)['info']
