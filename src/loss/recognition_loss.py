import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import str_filt


class CRNNRecognitionLoss(nn.Module):
    """CTC loss of a frozen CRNN reading the super-resolved image: how far its reading is from the label.

    The CRNN sees the image exactly as in the TextZoom evaluation (RGB channels, resized to 32 x 100,
    converted to grey). Its weights never change; the gradient only flows back into the SR network.
    """

    def __init__(self, crnn, converter):
        super().__init__()
        self.crnn = crnn.eval()
        for p in self.crnn.parameters():
            p.requires_grad_(False)
        self.converter = converter
        self.alphabet = set(converter.dict)
        self.ctc = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    def train(self, mode=True):
        return self  # the recognizer stays frozen whatever its owner does

    def forward(self, images_sr, labels):
        # cuDNN only backpropagates through an RNN in training mode. The CRNN's LSTMs have no
        # dropout, so their output is the same in both modes; BatchNorm stays in eval mode.
        for module in self.crnn.modules():
            if isinstance(module, nn.LSTM):
                module.train()
        images = F.interpolate(images_sr[:, :3], (32, 100), mode='bicubic')
        grey = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
        log_probs = self.crnn(grey).log_softmax(2)  # (frames, batch, classes)
        texts = [''.join(c for c in str_filt(label, 'lower') if c in self.alphabet) for label in labels]
        targets, target_lengths = self.converter.encode(texts)
        frames = torch.full((len(texts),), log_probs.size(0), dtype=torch.long)
        return self.ctc(log_probs, targets.long().to(log_probs.device), frames, target_lengths.long())
