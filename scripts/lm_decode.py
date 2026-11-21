#!/usr/bin/env python3
"""Beam search with a word n-gram LM over saved CTC log-probs (IMPROVEMENTS.md phase A2-A5).

  # 1. LM from training transcripts (dev: sessions 1-131, for tuning; final: 1-140)
  python scripts/lm_decode.py lm --cache /workspace/cache/v1 --split dev   --out /workspace/lm/dev
  python scripts/lm_decode.py lm --cache /workspace/cache/v1 --split final --out /workspace/lm/final
  # 2. tune alpha/beta on dev validation with a dev-split model and the dev LM
  python scripts/lm_decode.py tune --logprobs /workspace/runs/baseline_vs_dev/logprobs_val_best \
      --tokenizer /workspace/runs/baseline_vs_dev/tokenizer.model --lm /workspace/lm/dev
  # 3. score test once with the frozen settings
  python scripts/lm_decode.py decode --logprobs /workspace/runs/final_vs_final/logprobs_test_last \
      --tokenizer /workspace/runs/final_vs_final/tokenizer.model --lm /workspace/lm/final \
      --alpha A --beta B --beam 128 --oov-ref-split final

Runs in the separate decode venv (/workspace/venv-decode: pyctcdecode needs numpy < 2).
LM text is the normalised training transcripts in the model's own style (contractions kept);
scoring applies Whisper's normaliser afterwards. Words outside the LM's unigram list are not
forbidden, only penalised (pyctcdecode unk_score_offset), so test-only words remain possible.
"""
import argparse
import csv
import itertools
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import report, wnorm  # noqa: E402

KENLM_BIN = Path("/workspace/tools/kenlm/build/bin")


def train_texts(cache, split):
    with open(Path(cache) / "manifest.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if not r["dropped"] and r[f"{split}_split"] == "train"]
    return sorted({r["text"] for r in rows})


def cmd_lm(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    texts = train_texts(a.cache, a.split)
    (out / "text.txt").write_text("\n".join(texts) + "\n")
    with open(out / "text.txt") as fin, open(out / "lm.arpa", "w") as fout:
        subprocess.run([str(KENLM_BIN / "lmplz"), "-o", str(a.order), "--discount_fallback", "-S", "20%"],
                       stdin=fin, stdout=fout, stderr=subprocess.DEVNULL, check=True)
    subprocess.run([str(KENLM_BIN / "build_binary"), str(out / "lm.arpa"), str(out / "lm.bin")],
                   stdout=subprocess.DEVNULL, check=True)
    words = sorted({w for t in texts for w in t.split()})
    (out / "unigrams.txt").write_text("\n".join(words) + "\n")
    json.dump(dict(split=a.split, order=a.order, sentences=len(texts), word_types=len(words)),
              open(out / "lm.json", "w"), indent=1)
    print(f"{a.split} LM: {len(texts)} sentences, {len(words)} word types -> {out}", flush=True)


def labels_from_tokenizer(path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(path))
    return [""] + [sp.id_to_piece(i) for i in range(sp.get_piece_size())]


def load(stem):
    z = np.load(f"{stem}.npz")
    meta = list(csv.DictReader(open(f"{stem}.csv")))
    return [z[m["key"]].astype(np.float32) for m in meta], meta


_DEC = None


def _init(labels, lm_dir, alpha, beta):
    global _DEC
    from pyctcdecode import build_ctcdecoder
    lm = str(Path(lm_dir) / "lm.bin") if lm_dir else None
    uni = Path(lm_dir, "unigrams.txt").read_text().split() if lm_dir else None
    _DEC = build_ctcdecoder(labels, kenlm_model_path=lm, unigrams=uni, alpha=alpha, beta=beta)


def _decode(args):
    lp, beam, nbest = args
    beams = _DEC.decode_beams(lp, beam_width=beam)
    return beams[0][0], [(b[0], float(b[-1])) for b in beams[:nbest]]


def run_decoder(lps, labels, lm_dir, alpha, beta, beam, nbest=1, procs=24):
    with Pool(procs, initializer=_init, initargs=(labels, lm_dir, alpha, beta)) as pool:
        res = pool.map(_decode, [(lp, beam, nbest) for lp in lps], chunksize=8)
    return [r[0] for r in res], [r[1] for r in res]


def by_mode(meta, hyps, oov=frozenset()):
    out = {}
    for m in ["voiced", "silent"]:
        idx = [i for i, x in enumerate(meta) if x["mode"] == m]
        out[m] = report([meta[i]["ref"] for i in idx], [hyps[i] for i in idx], oov)
    return out


def oov_words(cache, ref_split, meta):
    """Test (or val) word types never seen in that split's training transcripts (Whisper-normalised)."""
    train = {w for t in train_texts(cache, ref_split) for w in wnorm(t).split()}
    return frozenset(w for m in meta for w in wnorm(m["ref"]).split() if w not in train)


def fmt(r):
    return (f"V {r['voiced']['wer_whisper']:.2%} (ours {r['voiced']['wer_ours']:.2%})  "
            f"S {r['silent']['wer_whisper']:.2%} (ours {r['silent']['wer_ours']:.2%})")


def cmd_tune(a):
    lps, meta = load(a.logprobs)
    labels = labels_from_tokenizer(a.tokenizer)
    log = []
    greedy = by_mode(meta, [m["greedy"] for m in meta])
    print(f"greedy                     {fmt(greedy)}", flush=True)
    for alpha, beta in itertools.product(a.alphas, a.betas):
        hyps, _ = run_decoder(lps, labels, a.lm, alpha, beta, a.beam, procs=a.procs)
        r = by_mode(meta, hyps)
        mean = (r["voiced"]["wer_whisper"] + r["silent"]["wer_whisper"]) / 2
        log.append(dict(alpha=alpha, beta=beta, beam=a.beam, mean=mean, **{m: r[m] for m in r}))
        print(f"alpha {alpha:4.2f} beta {beta:4.2f}  {fmt(r)}  mean {mean:.2%}", flush=True)
    best = min(log, key=lambda x: x["mean"])
    print(f"best: alpha {best['alpha']} beta {best['beta']} (mean Whisper WER {best['mean']:.2%})", flush=True)
    json.dump(dict(greedy=greedy, grid=log, best=best), open(f"{a.logprobs}_tune_{Path(a.lm).name}.json", "w"), indent=1)


def cmd_decode(a):
    lps, meta = load(a.logprobs)
    labels = labels_from_tokenizer(a.tokenizer)
    oov = oov_words(a.cache, a.oov_ref_split, meta)
    hyps, nb = run_decoder(lps, labels, a.lm, a.alpha, a.beta, a.beam, nbest=a.nbest, procs=a.procs)
    greedy = by_mode(meta, [m["greedy"] for m in meta], oov)
    r = by_mode(meta, hyps, oov)
    print(f"{len(oov)} OOV word types")
    print(f"greedy  {fmt(greedy)}  OOV recall V {greedy['voiced']['oov_recall']:.1%} S {greedy['silent']['oov_recall']:.1%}")
    print(f"beam+LM {fmt(r)}  OOV recall V {r['voiced']['oov_recall']:.1%} S {r['silent']['oov_recall']:.1%}")
    tag = f"{Path(a.lm).name if a.lm else 'nolm'}_a{a.alpha}_b{a.beta}_w{a.beam}"
    json.dump(dict(settings=vars(a), oov_types=len(oov), greedy=greedy, beam=r),
              open(f"{a.logprobs}_decode_{tag}.json", "w"), indent=1)
    with open(f"{a.logprobs}_decode_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "mode", "ref", "greedy", "hyp", "nbest"])
        for m, h, n in zip(meta, hyps, nb):
            w.writerow([m["key"], m["mode"], m["ref"], m["greedy"], h, json.dumps(n)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lm")
    p.add_argument("--cache", required=True)
    p.add_argument("--split", required=True, choices=["dev", "final"])
    p.add_argument("--out", required=True)
    p.add_argument("--order", type=int, default=4)
    for name in ("tune", "decode"):
        p = sub.add_parser(name)
        p.add_argument("--logprobs", required=True, help="stem of logprobs_<split>_<ckpt>.npz/.csv")
        p.add_argument("--tokenizer", required=True)
        p.add_argument("--lm", help="LM dir from the lm command (omit for beam search without LM)")
        p.add_argument("--beam", type=int, default=128)
        p.add_argument("--procs", type=int, default=24)
    sub.choices["tune"].add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.35, 0.5, 0.75, 1.0, 1.5])
    sub.choices["tune"].add_argument("--betas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 3.0])
    d = sub.choices["decode"]
    d.add_argument("--alpha", type=float, default=0.5)
    d.add_argument("--beta", type=float, default=1.0)
    d.add_argument("--nbest", type=int, default=10)
    d.add_argument("--cache", default="/workspace/cache/v1")
    d.add_argument("--oov-ref-split", default="final", choices=["dev", "final"])
    a = ap.parse_args()
    {"lm": cmd_lm, "tune": cmd_tune, "decode": cmd_decode}[a.cmd](a)


if __name__ == "__main__":
    main()
