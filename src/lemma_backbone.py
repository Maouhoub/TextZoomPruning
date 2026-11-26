"""Thinner and shallower versions of LEMMA's vision backbone (ResNet45 + transformer).

Section 22 showed that the backbone takes 78-87% of LEMMA's guidance time (GPU batch 64, CPU), while the
parts after it (position attention, language model, alignment) are cheap and must be kept: LEMMA's
guidance depends on its own reader's outputs. So the backbone is the part to prune.

The pruned backbone keeps the backbone's interface (512 features on the 8 x 32 grid), so everything after
it stays as released. Inside it:
- ResNet45: the output width of stages 1-4 and the middle width of every block (conv1 -> conv2) scale with
  `width`, rounded to a multiple of 8. Stage 5's output stays 512 (the interface).
- Transformer: `layers` of the 3 encoder layers are kept; d_model stays 512 (the interface) and the
  feed-forward width scales with `width` (multiple of 64). With 0 layers the transformer and its
  positional encoding are skipped.

The modules and parameter names follow LEMMA's own (lemma_net/resnet.py, backbone.py), so at width 1 with
3 layers the slim backbone loads LEMMA's backbone weights exactly (check_pruning.py check 7).
"""
import copy

import torch.nn as nn

from lemma_net.resnet import conv1x1, conv3x3
from lemma_net.transformer import PositionalEncoding, TransformerEncoder, TransformerEncoderLayer

STAGE_OUT = (32, 64, 128, 256, 512)  # ResNet45 output width per stage
BLOCKS = (3, 4, 6, 6, 3)
STRIDES = (2, 1, 2, 1, 1)
D_MODEL, HEADS, FFN, LAYERS = 512, 8, 2048, 3  # LEMMA's transformer (backbone.py)


def round_to(value, multiple):
    return max(multiple, int(round(value / multiple)) * multiple)


def slim_widths(width):
    """Stage outputs (the last stays 512) and block middles for a width multiplier."""
    outs = [round_to(c * width, 8) for c in STAGE_OUT[:-1]] + [STAGE_OUT[-1]]
    mids = [round_to(c * width, 8) for c in STAGE_OUT]
    return outs, mids


class SlimBlock(nn.Module):
    """LEMMA's BasicBlock (1x1 conv, then 3x3 conv with the stride) with its own middle width."""

    def __init__(self, inplanes, mid, planes, stride=1):
        super().__init__()
        self.conv1 = conv1x1(inplanes, mid)
        self.bn1 = nn.BatchNorm2d(mid)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class SlimResNet45(nn.Module):
    def __init__(self, width=1.0):
        super().__init__()
        outs, mids = slim_widths(width)
        self.outs, self.mids = outs, mids
        self.conv1 = nn.Conv2d(3, outs[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outs[0])
        self.relu = nn.ReLU(inplace=True)
        inplanes = outs[0]
        for stage in range(5):
            blocks = []
            for b in range(BLOCKS[stage]):
                blocks.append(SlimBlock(inplanes, mids[stage], outs[stage], STRIDES[stage] if b == 0 else 1))
                inplanes = outs[stage]
            setattr(self, f'layer{stage + 1}', nn.Sequential(*blocks))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        for stage in range(1, 6):
            x = getattr(self, f'layer{stage}')(x)
        return x


class SlimBackbone(nn.Module):
    """LEMMA's ResTranformer (backbone.py) with a slim ResNet45 and fewer, narrower encoder layers."""

    def __init__(self, width=1.0, layers=LAYERS):
        super().__init__()
        self.width, self.layers = width, layers
        self.resnet = SlimResNet45(width)
        self.ffn = round_to(FFN * width, 64)
        if layers > 0:
            self.pos_encoder = PositionalEncoding(D_MODEL, max_len=8 * 32)
            encoder_layer = TransformerEncoderLayer(d_model=D_MODEL, nhead=HEADS, dim_feedforward=self.ffn,
                                                    dropout=0.1, activation='relu')
            self.transformer = TransformerEncoder(encoder_layer, layers)

    def forward(self, images):
        feature = self.resnet(images)
        if self.layers == 0:
            return feature
        n, c, h, w = feature.shape
        feature = self.transformer(self.pos_encoder(feature.view(n, c, -1).permute(2, 0, 1)))
        return feature.permute(1, 2, 0).view(n, c, h, w)


def with_backbone(lemma, backbone):
    """A copy of LEMMA (lemma_teacher.LemmaSR) whose reader uses `backbone`; everything else as in `lemma`."""
    model = copy.deepcopy(lemma)
    model.net.position_prior.vision.backbone = backbone.to(next(lemma.parameters()).device)
    return model.eval()


def backbone_parameters(model):
    return sum(p.numel() for p in model.net.position_prior.vision.backbone.parameters())
