"""LEMMA (Guo et al., IJCAI 2023) as a frozen model: a distillation teacher and a speed reference.

The network is LEMMA's own code (lemma_net/). LEMMA's released file holds the weights of both its
branches (super-resolution and text guidance), so nothing else is downloaded and LEMMA is never
trained here. LemmaSR gives it the interface of our TSRN (align, reconstruct, forward), so the
fine-tuning, evaluation and timing code can use it unchanged.
"""
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

from lemma_net.lemma import LEMMA

CHARSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lemma_net', 'charset_36.txt')
TRAINING_ONLY = ('net.stn_head.', 'net.tps.')  # LEMMA's STN runs in training only, as in TSRN
GUIDANCE = ('net.position_prior.', 'net.guidanceGen.')  # the text-guidance branch
README_ACCURACY = {'aster': 66.93, 'moran': 63.60, 'crnn': 56.73}  # LEMMA's README, for the released file


def guidance_config():
    """LEMMA's `PositionAware` settings (its config/super_resolution.yaml) without the paths of the
    pre-trained ABINet parts: the released file already holds those weights."""
    return EasyDict({'PositionAware': {
        'dataset_max_length': 25, 'dataset_charset_path': CHARSET, 'model_vision_attention_mode': 'nearest',
        'vision': {'checkpoint': None, 'loss_weight': 1., 'attention': 'position', 'backbone': 'transformer',
                   'backbone_ln': 3, 'd_model': 512},
        'language': {'checkpoint': None, 'num_layers': 4, 'loss_weight': 1., 'detach': True,
                     'use_self_attn': False}}})


class LemmaSR(nn.Module):
    """LEMMA with our TSRN's interface: forward(LR image) returns only the SR image."""

    def __init__(self):
        super().__init__()
        # Same settings as LEMMA's main.py: STN, mask channel, 5 residual blocks.
        self.net = LEMMA(scale_factor=2, width=128, height=32, STN=True, mask=True, srb_nums=5,
                         cfg=guidance_config())

    def align(self, x):
        """Identity: LEMMA's STN only runs in training, and LEMMA is never trained here."""
        return x

    def reconstruct(self, x):
        return self.net(x)[0]

    def forward(self, x):
        return self.reconstruct(x)


class LemmaGuidance(nn.Module):
    """Only LEMMA's text-guidance branch, exactly as in LEMMA.forward, to time it on its own."""

    def __init__(self, lemma):
        super().__init__()
        self.lemma = lemma

    def forward(self, x):
        net = self.lemma.net
        upscaled = F.interpolate(x[:, :3], scale_factor=2, mode='bicubic', align_corners=True)
        guidance, _ = net.guidanceGen(net.position_prior(upscaled), net.block1(x))
        return guidance


def read_weights(checkpoint):
    """The network weights of LEMMA's released file.

    Tried first as plain data (safe loading). The file also stores LEMMA's training history; only if
    that history needs full unpickling is the file loaded that way (it comes from the link in
    LEMMA's official README).
    """
    try:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    except pickle.UnpicklingError as error:
        print(f'note: {os.path.basename(checkpoint)} cannot be read as plain data '
              f'({str(error).strip().splitlines()[0][:120]}...); loading it fully as the official LEMMA release')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    return {name.removeprefix('module.'): value for name, value in saved['state_dict_G'].items()}


def load_lemma(checkpoint, device):
    """The released LEMMA, frozen and in eval mode. Every weight must match the network (strict)."""
    model = LemmaSR()
    model.net.load_state_dict(read_weights(checkpoint))
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def parameter_counts(model):
    """Parameters used at inference (the training-only STN excluded), and the guidance branch's share."""
    used = {name: p.numel() for name, p in model.named_parameters() if not name.startswith(TRAINING_ONLY)}
    guidance = sum(n for name, n in used.items() if name.startswith(GUIDANCE))
    return {'inference': sum(used.values()), 'guidance_branch': guidance,
            'sr_branch': sum(used.values()) - guidance}
