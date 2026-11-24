#!/usr/bin/env python3
"""Save per-utterance CTC log-probabilities so decoding can be re-run without a GPU
(IMPROVEMENTS.md, phase A1).

  python scripts/dump_logprobs.py --run /workspace/runs/final_vs_final --ckpt last --split test
  python scripts/dump_logprobs.py --run /workspace/runs/baseline_vs_dev --ckpt best --split val

Writes <run>/logprobs_<split>_<ckpt>.npz with one float16 array (T', 1001) per utterance,
trimmed to its true output length, keyed "<mode>/<session>/<utt_id>"; plus
<run>/logprobs_<split>_<ckpt>.csv with key, mode, session, utt_id, ref (the reference text) and
greedy (the greedy decode, for checking a decoder reproduces the greedy WER at alpha=beta=0).
Class 0 is the CTC blank; class k is SentencePiece piece k-1 (see src/decode.py).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import BucketBatchSampler, EchoDataset, collate, load_rows, preload  # noqa: E402
from decode import Tokenizer, greedy_decode  # noqa: E402
from model import ResNet34CTC, build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="last", help="best | last | epochN | path to a .pt file")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--cache", help="override the cache path stored in the run config")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = yaml.safe_load(open(run / "config.yaml"))["cfg"]
    cache = Path(args.cache or cfg["cache"])
    ck_path = Path(args.ckpt) if args.ckpt.endswith(".pt") else run / f"{args.ckpt}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    amp = torch.bfloat16 if device.type == "cuda" else None

    tok = Tokenizer(run / "tokenizer.model")
    stats = json.load(open(run / "stats.json"))
    rows = load_rows(cache, cfg["split"], args.split, ["voiced", "silent"])
    if not rows:
        sys.exit(f"no {args.split} rows for split '{cfg['split']}'")
    arrays = preload(cache, rows, cfg.get("crop"))
    ds = EchoDataset(rows, arrays, tok, stats, train=False, aug={})
    sampler = BucketBatchSampler([r["n_frames"] for r in rows], args.batch_size, shuffle=False)
    dl = torch.utils.data.DataLoader(ds, collate_fn=collate, num_workers=4, batch_sampler=sampler)
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model = build_model(cfg["model"], ck["n_classes"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    order = [i for batch in sampler for i in batch]
    out, meta, k = {}, [], 0
    with torch.no_grad():
        for x, lens, _, _, _, _ in dl:
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                lp = model(x.to(device), lens)
            olens = ResNet34CTC.out_lengths(lens)
            greedy = greedy_decode(lp, olens, tok)
            lp = lp.float().cpu().numpy()
            for b in range(lp.shape[0]):
                r = rows[order[k]]
                key = f"{r['mode']}/{r['session']}/{r['utt_id']}"
                out[key] = lp[b, : int(olens[b])].astype(np.float16)
                meta.append(dict(key=key, mode=r["mode"], session=r["session"], utt_id=r["utt_id"],
                                 ref=r["text"], greedy=greedy[b]))
                k += 1

    stem = run / f"logprobs_{args.split}_{ck_path.stem}"
    np.savez_compressed(f"{stem}.npz", **out)
    meta.sort(key=lambda m: m["key"])
    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(meta[0]))
        w.writeheader()
        w.writerows(meta)
    print(f"{len(out)} utterances (epoch {ck['epoch']}) -> {stem}.npz / .csv", flush=True)


if __name__ == "__main__":
    main()
