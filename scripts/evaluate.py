#!/usr/bin/env python3
"""Evaluate a trained run with greedy CTC decoding (README section 9).

  python scripts/evaluate.py --run /workspace/runs/baseline_vs_final --ckpt last --split test
  python scripts/evaluate.py --run /workspace/runs/baseline_vs_dev --ckpt best --split val

Writes <run>/eval_<split>_<ckpt>.csv (one row per utterance: mode, session, id, ref, hyp, errors)
and <run>/eval_<split>_<ckpt>.json (WER per mode). Uses the run's own tokenizer and
normalisation statistics; reports voiced and silent separately.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import jiwer
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import norm_text  # noqa: E402
from data import BucketBatchSampler, EchoDataset, collate, load_rows, preload  # noqa: E402
from decode import Tokenizer, greedy_decode, wer  # noqa: E402
from model import ResNet34CTC  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("--ckpt", default="best", help="best | last | path to a .pt file")
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
    dl = torch.utils.data.DataLoader(ds, collate_fn=collate, num_workers=4,
                                     batch_sampler=BucketBatchSampler([r["n_frames"] for r in rows],
                                                                      args.batch_size, shuffle=False))
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model = ResNet34CTC(n_classes=ck["n_classes"], range_stride=cfg["model"]["range_stride"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    order = [i for batch in dl.batch_sampler for i in batch]
    hyps = []
    with torch.no_grad():
        for x, lens, _, _, _, _ in dl:
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                lp = model(x.to(device))
            hyps += greedy_decode(lp, ResNet34CTC.out_lengths(lens), tok)

    out_rows, by_mode = [], {}
    for i, h in zip(order, hyps):
        r = rows[i]
        m = jiwer.process_words(norm_text(r["text"]), norm_text(h) or "<empty>")
        out_rows.append(dict(mode=r["mode"], session=r["session"], utt_id=r["utt_id"], ref=r["text"], hyp=h,
                             ref_words=len(r["text"].split()), errors=m.substitutions + m.deletions + m.insertions))
        by_mode.setdefault(r["mode"], ([], []))
        by_mode[r["mode"]][0].append(r["text"])
        by_mode[r["mode"]][1].append(h)
    out_rows.sort(key=lambda r: (r["mode"], r["session"], r["utt_id"]))

    stem = run / f"eval_{args.split}_{ck_path.stem}"
    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    result = {f"wer_{m}": wer(refs, hs) for m, (refs, hs) in sorted(by_mode.items())}
    result.update(n_utts={m: len(v[0]) for m, v in by_mode.items()}, ckpt=str(ck_path), epoch=ck["epoch"],
                  split=args.split, train_split=cfg["split"], modes_trained=cfg["modes"])
    json.dump(result, open(f"{stem}.json", "w"), indent=1)
    print(json.dumps(result, indent=1))
    print(f"per-utterance hypotheses: {stem}.csv")


if __name__ == "__main__":
    main()
