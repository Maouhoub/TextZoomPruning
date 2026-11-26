"""Checks before designing a small reader for LEMMA (notebook section 22). No training.

LEMMA's reader (its ABINet vision and language models) is 97% of the computation of its text guidance
(spec: project/lemma_reader_spec.md). These checks run LEMMA's released model with parts of the reader
timed, removed or replaced, to decide what a small replacement must contain:

C1. Size, MACs and time of each part of the reader (ResNet45, transformer, position attention, language
    model) and of the alignment. Times come from cumulative prefixes of the guidance, timed in one
    session with LEMMA (GPU batch 1 and 64, CPU 1 thread batch 1); a part's time is the difference
    between two prefixes.
C2. LEMMA guided by the vision part alone: the vision logits and lengths replace the language ones.
C3. The reader's own word accuracy on the LR test images, for its vision part and its language part.
C4. LEMMA guided by the true words: one-hot letters at the scale of the reader's own logits, the true
    lengths, and the reader's own maps.

Rules fixed before the run:
- C1: the part with the largest share of the guidance time on CPU batch 1 and GPU batch 64 is the
  first to replace.
- C2: within 1 point of LEMMA on every reader -> the small reader needs no language model.
- C3 + C4: test points gained per point of reading accuracy, (C4 - LEMMA) / (100 - reader accuracy),
  gives how much reading accuracy the small reader may lose within the 2-point budget of the pass rule.
  This is a rough, straight-line estimate.

    python check_lemma_reader.py --lemma <LEMMA-release.pth> --lemma_results <lemma_results.json> --out_dir <folder>

Writes reader_checks.json after every check; a rerun skips the checks already in it.
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import evaluate_lemma
import lemma_teacher
import tsrn_experiment as exp
from distill_lemma_guidance import compare_with_lemma
from utils.util import str_filt

PARTS = ('ResNet45', 'transformer x3', 'position attention + classifier', 'language model x4', 'alignment')
MAX_LOSS_WITHOUT_LANGUAGE = 1.0  # points per reader (C2)
BUDGET = 2.0  # points below LEMMA allowed by the pass rule (section 21)
TOLERANCE = 1e-4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lemma', required=True, help="LEMMA's released checkpoint")
    parser.add_argument('--lemma_results', required=True, help='lemma_results.json from evaluate_lemma.py')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--config', default='config/super_resolution.yaml')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='MORAN supports at most 256')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--gpu_repeats', type=int, default=100)
    parser.add_argument('--cpu_repeats', type=int, default=20)
    return parser.parse_args()


# ---------------------------------------------------------------- LEMMA's reader, piece by piece

def upscale(x):
    """The reader's input, as in LEMMA.forward: the LR RGB channels, bicubic x2."""
    return F.interpolate(x[:, :3], scale_factor=2, mode='bicubic', align_corners=True)


def normalize(reader, images):
    mean = reader.mean.to(images.device)[..., None, None]
    std = reader.std.to(images.device)[..., None, None]
    return (images - mean) / std


def read(reader, images):
    """PositionAwareModule.forward, also keeping the vision part's own logits and lengths."""
    vision = reader.vision(normalize(reader, images))
    vision_lengths = vision['pt_lengths'].clamp(2, reader.max_length)
    language = reader.language(torch.softmax(vision['logits'], dim=-1), vision_lengths)
    attn_map = F.interpolate(vision['attn_scores'], scale_factor=2, mode='bicubic', align_corners=True)
    return {'attn_map': attn_map, 'vision_logits': vision['logits'], 'vision_lengths': vision_lengths,
            'text_logits': language['logits'], 'pt_lengths': language['pt_lengths'].clamp(2, reader.max_length)}


def as_lemma(outputs, labels=None):
    """What LEMMA's alignment receives: maps, language logits, language lengths."""
    return {key: outputs[key] for key in ('attn_map', 'text_logits', 'pt_lengths')}


def vision_only(outputs, labels=None):
    """C2: the vision part's logits and lengths in place of the language model's."""
    return {'attn_map': outputs['attn_map'], 'text_logits': outputs['vision_logits'],
            'pt_lengths': outputs['vision_lengths']}


class TrueWords:
    """C4: the true words as the reader's letters, one-hot at the scale of the reader's own logits (mean
    top score and mean other score on the slots it reads), with the true lengths and the reader's maps."""

    def __init__(self, charset):
        self.charset = charset

    def __call__(self, outputs, labels):
        logits = outputs['text_logits']
        batch, slots, classes = logits.shape
        valid = torch.arange(slots, device=logits.device)[None] < outputs['pt_lengths'][:, None]
        top = logits.max(dim=-1).values
        high = top[valid].mean()
        low = ((logits.sum(dim=-1) - top) / (classes - 1))[valid].mean()
        words = [str_filt(label, 'lower')[:slots - 1] for label in labels]  # room for the end token
        targets = torch.tensor([self.charset.get_labels(word, length=slots) for word in words], device=logits.device)
        letters = low + (high - low) * F.one_hot(targets, classes).to(logits.dtype)
        lengths = torch.tensor([len(word) + 1 for word in words], device=logits.device).clamp(2, slots)
        return {'attn_map': outputs['attn_map'], 'text_logits': letters, 'pt_lengths': lengths}


class GuidedLemma(nn.Module):
    """LEMMA whose guidance is built from a chosen version of the reader's outputs (`highlevel`)."""

    def __init__(self, lemma, highlevel):
        super().__init__()
        self.net = lemma.net
        self.highlevel = highlevel

    def forward(self, x, labels=None):
        features = self.net.block1(x)
        outputs = read(self.net.position_prior, upscale(x))
        guidance, _ = self.net.guidanceGen(self.highlevel(outputs, labels), features)
        return lemma_teacher.sr_branch(self.net, features, guidance)


class GuidancePrefix(nn.Module):
    """LEMMA's text guidance from the LR image up to and including one part (for C1)."""

    def __init__(self, lemma, last):
        super().__init__()
        assert last in PARTS, last
        self.net, self.last = lemma.net, last

    def forward(self, x):
        reader = self.net.position_prior
        if self.last == 'alignment':
            return lemma_teacher.guidance_map(self.net, x, self.net.block1(x))
        images = upscale(x)
        if self.last == 'language model x4':
            return reader(images)['text_logits']
        normalized = normalize(reader, images)
        if self.last == 'position attention + classifier':
            return reader.vision(normalized)['logits']
        if self.last == 'transformer x3':
            return reader.vision.backbone(normalized)
        return reader.vision.backbone.resnet(normalized)


def part_modules(lemma):
    reader = lemma.net.position_prior
    return {'ResNet45': [reader.vision.backbone.resnet],
            'transformer x3': [reader.vision.backbone.transformer],
            'position attention + classifier': [reader.vision.attention, reader.vision.cls],
            'language model x4': [reader.language],
            'alignment': [lemma.net.guidanceGen]}


def part_parameters(lemma):
    return {part: sum(p.numel() for module in modules for p in module.parameters())
            for part, modules in part_modules(lemma).items()}


def decode(logits, charset):
    """Greedy reading: the most likely class per slot, up to the first end token."""
    words = []
    for row in logits.argmax(dim=-1).tolist():
        letters = []
        for label in row:
            if label == charset.null_label:
                break
            letters.append(charset.label_to_char[label])
        words.append(''.join(letters))
    return words


# ---------------------------------------------------------------- the four checks

def check_c1(lemma, args):
    print('C1. size, MACs and time of each part (prefixes timed with LEMMA in one session)...', flush=True)
    params = part_parameters(lemma)
    prefixes = {part: GuidancePrefix(lemma, part) for part in PARTS}
    prefix_macs = {part: evaluate_lemma.aten_macs(prefix) for part, prefix in prefixes.items()}
    latency = evaluate_lemma.time_models({'LEMMA': lemma, **prefixes}, args)
    settings = list(latency['LEMMA'])
    guidance = latency['alignment']
    parts, previous_ms, previous_macs = {}, {key: 0.0 for key in settings}, 0
    for part in PARTS:
        macs = prefix_macs[part]
        parts[part] = {'parameters': params[part],
                       'macs': None if macs is None or previous_macs is None else macs - previous_macs,
                       'ms': {key: latency[part][key] - previous_ms[key] for key in settings}}
        parts[part]['share_of_guidance'] = {key: parts[part]['ms'][key] / guidance[key] for key in settings}
        previous_ms, previous_macs = latency[part], macs
    image_branch = {key: latency['LEMMA'][key] - guidance[key] for key in settings}
    first_to_replace = {}
    for setting in ('cpu1_b1', 'gpu_b64'):
        if setting in settings:  # no GPU: CPU only
            first_to_replace[setting] = max(PARTS, key=lambda part: parts[part]['ms'][setting])
    print(f"   {'part':34s} {'params':>11s} {'GMACs':>6s}  " + '  '.join(f'{key:>15s}' for key in settings))
    for part, value in parts.items():
        gmacs = '' if value['macs'] is None else f"{value['macs'] / 1e9:.3f}"
        print(f"   {part:34s} {value['parameters']:>11,} {gmacs:>6s}  " + '  '.join(
            f"{value['ms'][key]:7.2f} ms {100 * value['share_of_guidance'][key]:3.0f}%" for key in settings))
    print(f"   {'guidance (sum)':34s} {sum(params.values()):>11,} {'':6s}  "
          + '  '.join(f'{guidance[key]:7.2f} ms     ' for key in settings))
    print(f"   {'LEMMA':34s} {'':11s} {'':6s}  " + '  '.join(f"{latency['LEMMA'][key]:7.2f} ms     " for key in settings))
    print('   (a part = difference of two prefixes, so a small part can come out slightly negative at GPU batch 1)')
    print('   largest part: ' + ', '.join(f'{key} {part}' for key, part in first_to_replace.items()))
    return {'parts': parts, 'prefix_latency_ms': latency, 'prefix_macs': prefix_macs, 'guidance_ms': guidance,
            'image_branch_ms': image_branch, 'first_to_replace': first_to_replace, 'gpu': exp.gpu_name()}


def run_variant(name, model, test_loaders, textbase, recognizers, device, lemma_test, with_labels=False):
    test = exp.evaluate_test(model, test_loaders, textbase, recognizers, device, keep_predictions=True,
                             with_labels=with_labels)
    versus = compare_with_lemma(test, lemma_test)
    accuracy = test['average']['accuracy']
    for rec in exp.RECOGNIZERS:
        v = versus[rec]
        print(f"   {rec.upper():6s} {accuracy[rec]:.2f} (LEMMA {lemma_test['average']['accuracy'][rec]:.2f}, "
              f"{v['delta']:+.2f}, -{v['lost']}/+{v['gained']}, p={v['p']:.3g})")
    return {'name': name, 'test': test, 'accuracy': accuracy, 'versus_lemma': versus}


@torch.no_grad()
def check_c3(lemma, test_loaders, device):
    print("C3. the reader's own word accuracy on the LR test images...", flush=True)
    charset = lemma.net.position_prior.language.charset
    per_split = {}
    for split, loader in test_loaders.items():
        right = {'vision': 0, 'language': 0}
        images = 0
        for _, images_lr, labels in loader:
            outputs = read(lemma.net.position_prior, upscale(images_lr.to(device)))
            words = [str_filt(label, 'lower') for label in labels]
            for part, key in (('vision', 'vision_logits'), ('language', 'text_logits')):
                right[part] += sum(r == w for r, w in zip(decode(outputs[key], charset), words))
            images += len(words)
        per_split[split] = {**{part: 100.0 * count / images for part, count in right.items()}, 'images': images}
        print(f"   {split:6s} vision {per_split[split]['vision']:.2f}%, language {per_split[split]['language']:.2f}%")
    total = sum(r['images'] for r in per_split.values())
    average = {part: sum(r[part] * r['images'] for r in per_split.values()) / total for part in ('vision', 'language')}
    print(f"   average vision {average['vision']:.2f}%, language {average['language']:.2f}%")
    return {'per_split': per_split, 'average': average}


def summary(results, lemma_accuracy):
    c1, c2, c3, c4 = (results.get(key) for key in ('c1', 'c2', 'c3', 'c4'))
    decisions = {}
    print()
    print('===== what the checks say')
    if c1:
        decisions['first_to_replace'] = c1['first_to_replace']
        print('C1 replace first: ' + ', '.join(f'{key}: {part}' for key, part in c1['first_to_replace'].items()))
    if c2:
        losses = {rec: lemma_accuracy[rec] - c2['accuracy'][rec] for rec in exp.RECOGNIZERS}
        no_language = all(loss <= MAX_LOSS_WITHOUT_LANGUAGE for loss in losses.values())
        decisions['language_model_needed'] = not no_language
        print('C2 without the language model: ' + ', '.join(f'{rec.upper()} {-loss:+.2f}' for rec, loss in losses.items())
              + f" -> {'no language model needed' if no_language else 'the small reader needs a language part'}")
    if c3:
        print(f"C3 reader reads {c3['average']['language']:.2f}% of LR test words right "
              f"({c3['average']['vision']:.2f}% without the language model)")
    if c3 and c4:
        reading = c3['average']['language']
        slopes = {rec: (c4['accuracy'][rec] - lemma_accuracy[rec]) / (100.0 - reading) for rec in exp.RECOGNIZERS}
        allowed = {rec: (BUDGET / slope if slope > 0 else None) for rec, slope in slopes.items()}
        decisions['points_per_reading_point'] = slopes
        decisions['reading_loss_allowed'] = allowed
        print('C4 true words: ' + ', '.join(f"{rec.upper()} {c4['accuracy'][rec]:.2f} ({c4['accuracy'][rec] - lemma_accuracy[rec]:+.2f})"
                                            for rec in exp.RECOGNIZERS))
        print('   test points per point of reading accuracy: ' + ', '.join(f'{rec.upper()} {s:.3f}' for rec, s in slopes.items()))
        print(f'   reading accuracy the small reader may lose for {BUDGET:g} points: ' + ', '.join(
            f"{rec.upper()} {'no limit found' if a is None else f'{a:.1f} points'}" for rec, a in allowed.items())
              + ' (rough, straight-line estimate)')
    return decisions


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'reader_checks.json')
    results = exp.read_json(path) if os.path.exists(path) else {'args': vars(args), 'environment': exp.environment_info()}
    save = lambda: exp.write_json(path, results)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = exp.load_config(args.config)
    textbase = exp.make_textbase(config, args.eval_batch_size)
    lemma = lemma_teacher.load_lemma(args.lemma, device)
    lemma_test = exp.read_json(args.lemma_results)['test']
    lemma_accuracy = lemma_test['average']['accuracy']
    test_loaders = exp.make_test_loaders(config.TRAIN, args.eval_batch_size)

    with torch.no_grad():  # the pieces used below must rebuild LEMMA exactly
        _, images_lr, _ = next(iter(test_loaders['easy']))
        x = images_lr[:32].to(device)
        difference = (GuidedLemma(lemma, as_lemma)(x) - lemma(x)).abs().max().item()
    print(f'0. LEMMA rebuilt from its pieces on 32 test images: largest difference {difference:.1e}')
    if difference > TOLERANCE:
        raise RuntimeError('the pieces do not reproduce LEMMA; the checks would not be valid')

    if 'c1' not in results:
        results['c1'] = check_c1(lemma, args)
        save()
    recognizers = None
    for key, name, highlevel, with_labels in (('c2', 'vision part only', vision_only, False),
                                              ('c4', 'true words', TrueWords(lemma.net.position_prior.language.charset), True)):
        if key in results:
            continue
        if recognizers is None:
            recognizers = {rec: textbase.load_recognizer(rec) for rec in exp.RECOGNIZERS}
        label = 'C2. LEMMA guided by the vision part only' if key == 'c2' else 'C4. LEMMA guided by the true words'
        print(f'{label} (test, McNemar vs LEMMA)...', flush=True)
        results[key] = run_variant(name, GuidedLemma(lemma, highlevel), test_loaders, textbase, recognizers, device,
                                   lemma_test, with_labels)
        save()
    if 'c3' not in results:
        results['c3'] = check_c3(lemma, test_loaders, device)
        save()
    results['decisions'] = summary(results, lemma_accuracy)
    save()
    print(f'saved {path}')


if __name__ == '__main__':
    main()
