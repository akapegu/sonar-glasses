"""Dataset, augmentation and batching (README section 6).

- rows come from <cache>/manifest.csv; a split is chosen with split_kind ("dev" | "final") and
  the value ("train" | "val" | "test"), restricted to the requested modes
- arrays are float16 (4, 80, T) on disk and are preloaded into RAM (the full corpus is ~11 GB)
- normalisation: per-channel (x - mean) / std with statistics from the training split only
- training-time augmentation (never on val/test):
    concatenation: with prob concat_p join 2..concat_max random utterances, capped at max_frames
    SpecAugment: range_masks bands of up to range_mask_max bins and time_masks stretches of up to
    time_mask_max frames, set to 0 after normalisation
  EchoSpeech (CHI '23, section 5.5) options, all off by default:
    concat_consecutive: join utterances recorded in a row instead of random ones
    concat_gap_max: still gap of 0..N frames between joined utterances
    scale: multiply the sample by U(1 - scale, 1 + scale)
    shift_p / shift_max: with prob shift_p pad 0..shift_max still frames on each side
- crossCon targets (step 5b, optional): one speech-feature vector per CTC output step for each
  training row; the augmentations' concatenation, gaps and shift padding are tracked so each output
  step of the final sample gets the target of the frame at its centre (none in gaps / padding)
"""
import math
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_rows(cache: Path, split_kind: str, split: str, modes):
    rows = []
    with open(Path(cache) / "manifest.csv") as fh:
        for r in csv.DictReader(fh):
            if r["dropped"] or r["mode"] not in modes:
                continue
            if r[f"{split_kind}_split"] != split:
                continue
            r["n_frames"] = int(r["n_frames"])
            r["session"] = int(r["session"])
            rows.append(r)
    rows.sort(key=lambda r: (r["mode"], r["session"], r["utt_id"]))
    return rows


def cache_crop(cache: Path):
    """Range bins stored in a cache. Caches built before the crop was recorded hold 226:306."""
    try:
        crop = json.load(open(Path(cache) / "summary.json")).get("crop")
    except OSError:
        crop = None
    return tuple(crop) if crop else (226, 306)


def preload(cache: Path, rows, crop=None, threads: int = 16):
    """Load the arrays; with crop=(first, end) in absolute range bins, keep only those bins
    (must lie inside the cache's own crop)."""
    lo = hi = None
    if crop:
        c0, c1 = cache_crop(cache)
        assert c0 <= crop[0] < crop[1] <= c1, f"crop {crop} is outside the cache's bins {c0}:{c1}"
        lo, hi = crop[0] - c0, crop[1] - c0

    def one(r):
        a = np.load(Path(cache) / r["path"])
        return np.ascontiguousarray(a[:, lo:hi, :]) if crop else a
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(one, rows))


def compute_stats(arrays, max_utts: int = 4000, seed: int = 0):
    """Per-channel mean / std (float64 accumulation) over training utterances."""
    rng = random.Random(seed)
    idx = list(range(len(arrays)))
    rng.shuffle(idx)
    s = np.zeros(4)
    ss = np.zeros(4)
    n = 0
    for i in idx[:max_utts]:
        x = arrays[i].astype(np.float64)
        s += x.sum(axis=(1, 2))
        ss += (x ** 2).sum(axis=(1, 2))
        n += x.shape[1] * x.shape[2]
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-12))
    return dict(mean=mean.tolist(), std=std.tolist(), n_values_per_channel=int(n), n_utts=min(max_utts, len(idx)))


class EchoDataset(Dataset):
    def __init__(self, rows, arrays, tokenizer, stats, train: bool, aug: dict, targets=None):
        self.rows, self.arrays, self.tok, self.train, self.aug = rows, arrays, tokenizer, train, aug
        self.targets = targets  # None, or a list aligned with rows of (T', D) arrays / None
        self.target_dim = next((t.shape[1] for t in targets if t is not None), 0) if targets else 0
        self.mean = np.asarray(stats["mean"], np.float32)[:, None, None]
        self.std = np.asarray(stats["std"], np.float32)[:, None, None]

    def __len__(self):
        return len(self.rows)

    def _next_in_session(self, j):
        """Index of the utterance recorded right after j (rows are sorted by mode, session, id), or None."""
        k = j + 1
        if k < len(self.rows) and self.rows[k]["mode"] == self.rows[j]["mode"] \
                and self.rows[k]["session"] == self.rows[j]["session"]:
            return k
        return None

    def _concat_ids(self, i):
        ids = [i]
        a = self.aug
        if self.train and random.random() < a.get("concat_p", 0.0):
            k = random.randint(2, a.get("concat_max", 3))
            total = self.rows[i]["n_frames"]
            for _ in range(k - 1):
                if a.get("concat_consecutive"):  # EchoSpeech 5.5.1: utterances recorded in a row
                    j = self._next_in_session(ids[-1])
                    if j is None:
                        break
                else:
                    j = random.randrange(len(self.rows))
                if total + self.rows[j]["n_frames"] + a.get("concat_gap_max", 0) > a.get("max_frames", 3000):
                    break
                ids.append(j)
                total += self.rows[j]["n_frames"]
        return ids

    def _join(self, ids):
        """Concatenate along time; with concat_gap_max > 0 insert a still (all-zero) gap of
        0..concat_gap_max frames between utterances, like the zero-masked pauses in the recordings."""
        parts, gaps = [self.arrays[ids[0]]], []
        for j in ids[1:]:
            g = random.randint(0, self.aug.get("concat_gap_max", 0))
            gaps.append(g)
            if g:
                parts.append(np.zeros((*parts[0].shape[:2], g), parts[0].dtype))
            parts.append(self.arrays[j])
        return np.concatenate(parts, axis=2).astype(np.float32), gaps

    def _shift_pad(self, x):
        """EchoSpeech 5.5.3: place the sample at a random time offset inside still padding
        (0..shift_max frames on each side), counted in the input length so the model learns blank there."""
        a = self.aug
        if random.random() >= a.get("shift_p", 0.0):
            return x, 0
        room = max(0, a.get("max_frames", 3000) - x.shape[2])
        left = random.randint(0, min(a.get("shift_max", 0), room))
        right = random.randint(0, min(a.get("shift_max", 0), room - left))
        return np.pad(x, ((0, 0), (0, 0), (left, right))), left

    def _target_track(self, ids, gaps, left, n_t):
        """(T', D) targets and a (T',) validity mask for a sample of n_t frames built from ids."""
        n_out = math.ceil(n_t / 16)
        tgt = np.zeros((n_out, self.target_dim), np.float16)
        valid = np.zeros(n_out, bool)
        starts, pos = [], left
        for k, j in enumerate(ids):
            if k:
                pos += gaps[k - 1]
            starts.append(pos)
            pos += self.rows[j]["n_frames"]
        for t in range(n_out):
            f = 16 * t + 8
            for k, j in enumerate(ids):
                lf = f - starts[k]
                if 0 <= lf < self.rows[j]["n_frames"]:
                    tg = self.targets[j]
                    if tg is not None:
                        tgt[t] = tg[min(lf // 16, len(tg) - 1)]
                        valid[t] = True
                    break
        return torch.from_numpy(tgt), torch.from_numpy(valid)

    def _specaug(self, x):
        a = self.aug
        _, n_bins, n_t = x.shape
        for _ in range(a.get("range_masks", 0)):
            w = random.randint(0, a.get("range_mask_max", 0))
            if w:
                f0 = random.randint(0, n_bins - w)
                x[:, f0:f0 + w, :] = 0
        for _ in range(a.get("time_masks", 0)):
            w = random.randint(0, min(a.get("time_mask_max", 0), n_t // 4))
            if w:
                t0 = random.randint(0, n_t - w)
                x[:, :, t0:t0 + w] = 0
        return x

    def __getitem__(self, i):
        ids = self._concat_ids(i)
        gaps, left = [0] * (len(ids) - 1), 0
        if self.train:
            x, gaps = self._join(ids)
        else:
            x = np.concatenate([self.arrays[j] for j in ids], axis=2).astype(np.float32)
        if self.train:
            s = self.aug.get("scale", 0.0)  # EchoSpeech 5.5.2: multiply by U(1 - s, 1 + s)
            if s:
                x *= random.uniform(1 - s, 1 + s)
            x, left = self._shift_pad(x)  # still padding is raw 0 (no motion), normalised like the rest
        x = (x - self.mean) / self.std
        if self.train:
            x = self._specaug(x)
        text = " ".join(self.rows[j]["text"] for j in ids)
        y = self.tok.encode(text)
        item = (torch.from_numpy(x), torch.tensor(y, dtype=torch.long), text, self.rows[i]["mode"])
        if self.targets is not None:
            item += (self._target_track(ids, gaps, left, x.shape[2]),)
        return item


def collate(batch):
    """-> x, lengths, targets, target_lengths, texts, modes [, crossCon targets (B, T', D), valid (B, T')]"""
    xs, ys, texts, modes = zip(*[b[:4] for b in batch])
    lengths = torch.tensor([x.shape[2] for x in xs], dtype=torch.long)
    out = torch.zeros(len(xs), xs[0].shape[0], xs[0].shape[1], int(lengths.max()))
    for b, x in enumerate(xs):
        out[b, :, :, : x.shape[2]] = x
    targets = torch.cat(ys)
    target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    if len(batch[0]) == 5:
        tr = [b[4] for b in batch]
        n_out = max(t.shape[0] for t, _ in tr)
        tgt = torch.zeros(len(tr), n_out, tr[0][0].shape[1], dtype=torch.float16)
        valid = torch.zeros(len(tr), n_out, dtype=torch.bool)
        for b, (t, v) in enumerate(tr):
            tgt[b, : t.shape[0]] = t
            valid[b, : v.shape[0]] = v
        return out, lengths, targets, target_lengths, list(texts), list(modes), tgt, valid
    return out, lengths, targets, target_lengths, list(texts), list(modes)


class BucketBatchSampler(torch.utils.data.Sampler):
    """Shuffled batches of similar-length utterances: shuffle, cut into pools of
    `pool` batches, sort each pool by length, batch it, shuffle the batches."""

    def __init__(self, lengths, batch_size, pool=50, shuffle=True, seed=0):
        self.lengths, self.bs, self.pool, self.shuffle = list(lengths), batch_size, pool, shuffle
        self.epoch, self.seed = 0, seed

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)
        batches = []
        chunk = self.bs * self.pool
        for s in range(0, len(idx), chunk):
            part = sorted(idx[s:s + chunk], key=lambda i: self.lengths[i])
            batches += [part[k:k + self.bs] for k in range(0, len(part), self.bs)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)
