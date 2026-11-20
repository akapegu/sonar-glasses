"""Dataset, augmentation and batching (README section 6).

- rows come from <cache>/manifest.csv; a split is chosen with split_kind ("dev" | "final") and
  the value ("train" | "val" | "test"), restricted to the requested modes
- arrays are float16 (4, 80, T) on disk and are preloaded into RAM (the full corpus is ~11 GB)
- normalisation: per-channel (x - mean) / std with statistics from the training split only
- training-time augmentation (never on val/test):
    concatenation: with prob concat_p join 2..concat_max random utterances, capped at max_frames
    SpecAugment: range_masks bands of up to range_mask_max bins and time_masks stretches of up to
    time_mask_max frames, set to 0 after normalisation
"""
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


def preload(cache: Path, rows, threads: int = 16):
    def one(r):
        return np.load(Path(cache) / r["path"])
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
    def __init__(self, rows, arrays, tokenizer, stats, train: bool, aug: dict):
        self.rows, self.arrays, self.tok, self.train, self.aug = rows, arrays, tokenizer, train, aug
        self.mean = np.asarray(stats["mean"], np.float32)[:, None, None]
        self.std = np.asarray(stats["std"], np.float32)[:, None, None]

    def __len__(self):
        return len(self.rows)

    def _concat_ids(self, i):
        ids = [i]
        a = self.aug
        if self.train and random.random() < a.get("concat_p", 0.0):
            k = random.randint(2, a.get("concat_max", 3))
            total = self.rows[i]["n_frames"]
            for _ in range(k - 1):
                j = random.randrange(len(self.rows))
                if total + self.rows[j]["n_frames"] > a.get("max_frames", 3000):
                    break
                ids.append(j)
                total += self.rows[j]["n_frames"]
        return ids

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
        x = np.concatenate([self.arrays[j] for j in ids], axis=2).astype(np.float32)
        x = (x - self.mean) / self.std
        if self.train:
            x = self._specaug(x)
        text = " ".join(self.rows[j]["text"] for j in ids)
        y = self.tok.encode(text)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), text, self.rows[i]["mode"]


def collate(batch):
    xs, ys, texts, modes = zip(*batch)
    lengths = torch.tensor([x.shape[2] for x in xs], dtype=torch.long)
    out = torch.zeros(len(xs), xs[0].shape[0], xs[0].shape[1], int(lengths.max()))
    for b, x in enumerate(xs):
        out[b, :, :, : x.shape[2]] = x
    targets = torch.cat(ys)
    target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
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
