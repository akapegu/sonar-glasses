#!/usr/bin/env python3
"""Step 5b: crossCon targets, one speech-feature vector per CTC output step (12.5 Hz).

  /workspace/venv/bin/python experiments/07_transformer_crosscon/make_targets.py \
      --cache /workspace/cache/v1 --audio /workspace/cache/audio16k --out /workspace/cache/crosscon_h9 \
      --align-run /workspace/runs/baseline_vs_dev --layer 9

1. Voiced utterances: HuBERT-base (facebook/hubert-base-ls960) hidden layer `--layer` on the
   16 kHz audio from extract_audio.py (50 Hz), mean-pooled over groups of 4 frames to 12.5 Hz
   and padded/trimmed to the echo model's output length, ceil(n_frames / 16).
2. Silent utterances: the same sentence's voiced take (same text; the take from the nearest
   session if there are several) is aligned to the silent take with DTW on the echo model's
   encoder features (`--align-run`'s best checkpoint, 512-d per output step, cosine distance),
   and each silent step borrows the HuBERT vector of the voiced step it maps to.
Writes <out>/<echo path>.npy (float16, (T', 768)) and <out>/targets.csv (path, source, dtw_cost).
Only training-split utterances of the given --split get targets (validation and test never do).
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def out_len(n):
    return math.ceil(n / 16)


def hubert_targets(rows, audio_dir, out, layer, device):
    from transformers import HubertModel
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    feats = {}
    for i, r in enumerate(rows):
        dst = Path(out) / r["path"]
        if dst.exists():  # resumable
            feats[r["path"]] = np.load(dst)
            continue
        y = torch.from_numpy(np.load(Path(audio_dir) / r["path"]).astype(np.float32))[None].to(device)
        y = (y - y.mean()) / (y.std() + 1e-7)  # HuBERT expects zero-mean unit-variance input
        with torch.no_grad():
            h = model(y, output_hidden_states=True).hidden_states[layer][0]  # (T50, 768)
        T = out_len(int(r["n_frames"]))
        h = h[: 4 * T]
        if h.shape[0] < 4 * T:
            h = torch.cat([h, h[-1:].expand(4 * T - h.shape[0], -1)])
        f = h.view(T, 4, -1).mean(1).cpu().numpy().astype(np.float16)
        feats[r["path"]] = f
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, f)
        if (i + 1) % 1000 == 0:
            print(f"  hubert {i + 1}/{len(rows)}", flush=True)
    return feats


def encoder_features(run, cache, rows, device):
    """512-d encoder output per CTC step from a trained ResNet-34 (for DTW only)."""
    import yaml
    from data import load_rows  # noqa: F401  (keeps the import path consistent with the scripts)
    from model import build_model
    ck = torch.load(Path(run) / "best.pt", map_location=device, weights_only=False)
    cfg = ck.get("cfg") or yaml.safe_load(open(Path(run) / "config.yaml"))["cfg"]
    model = build_model(cfg["model"], ck["n_classes"]).to(device).eval()
    model.load_state_dict(ck["model"])
    import json
    stats = json.load(open(Path(run) / "stats.json"))
    mean = np.array(stats["mean"], np.float32)[:, None, None]
    std = np.array(stats["std"], np.float32)[:, None, None]
    out = {}
    for r in rows:
        x = np.load(Path(cache) / r["path"]).astype(np.float32)  # (4, bins, T)
        x = (x - mean) / std
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h = model.encoder(torch.from_numpy(x)[None].to(device)).mean(dim=2)[0].T.float()  # (T', 512)
        h = torch.nn.functional.normalize(h, dim=-1)
        out[r["path"]] = h.cpu().numpy()
    return out


def dtw_path(a, b):
    """DTW on cosine distance between unit vectors a (n, d) and b (m, d); returns, for each i in a,
    the matched index in b (the middle of its matched run), and the mean path cost."""
    C = 1.0 - a @ b.T
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    match = {}
    for p, q in path:
        match.setdefault(p, []).append(q)
    return np.array([int(np.median(match[p])) for p in range(n)]), float(C[tuple(zip(*path))].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--align-run", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "final"])
    ap.add_argument("--layer", type=int, default=9)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [r for r in csv.DictReader(open(Path(a.cache) / "manifest.csv")) if r["path"]]
    key = f"{a.split}_split"
    train = [r for r in rows if r[key] == "train"]
    voiced = [r for r in train if r["mode"] == "voiced"]
    silent = [r for r in train if r["mode"] == "silent"]
    print(f"{len(voiced)} voiced and {len(silent)} silent training utterances ({a.split} split)", flush=True)

    feats = hubert_targets(voiced, a.audio, a.out, a.layer, device)
    log = [dict(path=r["path"], source="own_audio", dtw_cost="") for r in voiced]

    by_text = {}
    for r in voiced:
        by_text.setdefault(r["text"], []).append(r)
    pairs = []
    for r in silent:
        cands = by_text.get(r["text"])
        if cands:
            pairs.append((r, min(cands, key=lambda v: abs(int(v["session"]) - int(r["session"])))))
    print(f"{len(pairs)} of {len(silent)} silent utterances have a voiced take of the same sentence", flush=True)
    enc = encoder_features(a.align_run, a.cache, [p for pr in pairs for p in pr], device)
    for i, (s, v) in enumerate(pairs):
        idx, cost = dtw_path(enc[s["path"]], enc[v["path"]])
        T = out_len(int(s["n_frames"]))
        f = feats[v["path"]][np.clip(idx, 0, len(feats[v["path"]]) - 1)][:T]
        dst = Path(a.out) / s["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, f.astype(np.float16))
        log.append(dict(path=s["path"], source=v["path"], dtw_cost=f"{cost:.4f}"))
        if (i + 1) % 1000 == 0:
            print(f"  dtw {i + 1}/{len(pairs)}", flush=True)
    with open(Path(a.out) / "targets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "source", "dtw_cost"])
        w.writeheader()
        w.writerows(log)
    print(f"done: {len(log)} target files in {a.out}", flush=True)


if __name__ == "__main__":
    main()
