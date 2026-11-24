#!/usr/bin/env python3
"""Where do word deletions happen? Sentence position and speaking rate, on a decode CSV.

  .venv/bin/python experiments/06_phoneme_analysis/deletion_analysis.py \
      --csv <run>/logprobs_test_last_decode_final_ext_a0.75_b3.0_w128.csv --cols greedy hyp \
      --data data --tokenizer <run>/tokenizer.model

Position. When an output is one word short and the neighbouring words differ, deleting either
word costs the same, and the alignment's tie-break decides where the deletion lands. jiwer puts
it as late as possible, which piles deletions onto the last word. This script aligns twice
(ties broken toward late and toward early deletions) and counts a deletion as "clean" only when
both alignments agree; the rest ("merge") are drops next to a wrong word. It also reports two
alignment-free checks: how often the reference's last word appears among the output's last two
words, and how many too-short outputs end on the reference's second-to-last word (truncation).

Speaking rate. Utterance durations come from the raw transcript.json files (data/*/..._eval/
session_*/transcript.json). Segments are back-to-back, so a duration includes the pause before
the sentence. Reports S / D (clean, merge) / I by quintile of words per second and of tokens per
CTC output step (16x downsampling at 200 fps), and how tokens per step relates to word rarity
(tokens per word).
"""
import argparse
import csv
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from whisper_normalizers.english import EnglishTextNormalizer  # noqa: E402

NORM = EnglishTextNormalizer()
FUNCTION_WORDS = set(
    "a an the and or but if of in on at to for from by with as is are was were be been am do does did i you he "
    "she it we they me him her us them my your his its our their this that these those not no so than then there "
    "here what who how when where why can will would could should just all some any up out about into over".split())


def table(r, h):
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d


def backtrace(r, h, d, late):
    """Deleted reference indices, substitution and insertion counts; late=True puts tied deletions last."""
    i, j, dele, S, I = len(r), len(h), set(), 0, 0
    while i or j:
        diag = i and j and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1])
        up = i and d[i][j] == d[i - 1][j] + 1
        if late and up:
            dele.add(i - 1); i -= 1
        elif diag:
            S += r[i - 1] != h[j - 1]; i -= 1; j -= 1
        elif up:
            dele.add(i - 1); i -= 1
        else:
            I += 1; j -= 1
    return dele, S, I


def durations(data):
    out = {}
    for f in glob.glob(f"{data}/*/SoniSpeech_*_eval/session_*/transcript.json"):
        mode = "silent" if "silent_eval" in f else "voiced"
        ses = int(re.search(r"session_(\d+)", f).group(1))
        for uid, u in json.load(open(f)).items():
            out[f"{mode}/{ses}/{uid}"] = u["end"] - u["start"]
    return out


def analyse(rows, col, mode, dur, sp):
    pos = {k: [0, 0, 0] for k in ("first", "middle", "last")}  # n, late-tie deletions, clean deletions
    wtype = {(p, t): [0, 0] for p in pos for t in ("function", "content")}
    last_kept = n_sent = short = ends_ok = ends_trunc = 0
    utts = []
    for r in rows:
        if r["mode"] != mode:
            continue
        ref, hyp = NORM(r["ref"]).split(), NORM(r[col]).split()
        if len(ref) < 3:
            continue
        d = table(ref, hyp)
        late, S, I = backtrace(ref, hyp, d, True)
        early, _, _ = backtrace(ref, hyp, d, False)
        clean = late & early
        for i, w in enumerate(ref):
            p = "first" if i == 0 else "last" if i == len(ref) - 1 else "middle"
            pos[p][0] += 1; pos[p][1] += i in late; pos[p][2] += i in clean
            t = wtype[(p, "function" if w in FUNCTION_WORDS else "content")]
            t[0] += 1; t[1] += i in clean
        n_sent += 1
        last_kept += ref[-1] in hyp[-2:]
        if len(hyp) < len(ref):
            short += 1
            ends_ok += bool(hyp) and hyp[-1] == ref[-1]
            ends_trunc += bool(hyp) and hyp[-1] == ref[-2]
        if r["key"] in dur:
            T = round(dur[r["key"]] * 200)
            toks = len(sp.encode(r["ref"])) if sp else float("nan")
            utts.append(dict(wps=len(ref) / dur[r["key"]], tps=toks / math.ceil(T / 16), tpw=toks / len(ref),
                             n=len(ref), S=S, D=len(late), clean=len(clean), merge=len(late) - len(clean), I=I))
    print(f"\n== {col} / {mode}")
    print("  deletion rate by position:  " + "   ".join(
        f"{p}: jiwer-style {v[1] / v[0]:.1%}, clean {v[2] / v[0]:.1%}" for p, v in pos.items()))
    print("  clean deletions by position x word type:  " + "   ".join(
        f"{p}/{t} {v[1] / v[0]:.1%}" for (p, t), v in wtype.items()))
    print(f"  last reference word among the output's last 2 words: {last_kept / n_sent:.1%}; "
          f"too-short outputs {short}: end on the last word {ends_ok}, on the 2nd-to-last {ends_trunc}")
    res = dict(position=pos, last_word_kept=last_kept / n_sent, too_short=short, ends_ok=ends_ok, ends_trunc=ends_trunc)
    if not utts:
        return res
    for key, label in [("wps", "words/s"), ("tps", "tokens/output step")]:
        x = np.array([u[key] for u in utts])
        if np.isnan(x).all():
            continue
        qs = np.quantile(x, [0, .2, .4, .6, .8, 1])
        print(f"  by {label} quintile:   WER     S      D   clean  merge    I")
        for a, b in zip(qs[:-1], qs[1:]):
            g = [u for u in utts if a <= u[key] <= b]
            n = sum(u["n"] for u in g)
            f = {k: sum(u[k] for u in g) / n for k in ("S", "D", "clean", "merge", "I")}
            print(f"    {a:5.2f}-{b:5.2f}  {f['S'] + f['D'] + f['I']:6.1%} {f['S']:5.1%} {f['D']:5.1%} "
                  f"{f['clean']:5.1%} {f['merge']:5.1%} {f['I']:5.1%}")
        dr = np.array([u["D"] / u["n"] for u in utts])
        sr = np.array([u["S"] / u["n"] for u in utts])
        print(f"    correlation with per-sentence deletion rate {np.corrcoef(x, dr)[0, 1]:+.2f}, "
              f"substitution rate {np.corrcoef(x, sr)[0, 1]:+.2f}")
    if sp:
        tpw, tps, wps = (np.array([u[k] for u in utts]) for k in ("tpw", "tps", "wps"))
        dr = np.array([u["D"] / u["n"] for u in utts])
        X = np.c_[np.ones_like(tpw), (tpw - tpw.mean()) / tpw.std(), (wps - wps.mean()) / wps.std()]
        coef = np.linalg.lstsq(X, dr, rcond=None)[0][1:] * 100
        print(f"  tokens/step vs tokens/word correlation {np.corrcoef(tps, tpw)[0, 1]:+.2f}; deletion rate per SD: "
              f"rarity (tokens/word) {coef[0]:+.2f} pts, speaking rate {coef[1]:+.2f} pts; max tokens/step {tps.max():.2f}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cols", nargs="+", default=["greedy", "hyp"])
    ap.add_argument("--data", default=str(ROOT / "data"), help="raw dataset root with the eval transcript.json files")
    ap.add_argument("--tokenizer", help="SentencePiece model, for tokens per output step")
    a = ap.parse_args()
    sp = None
    if a.tokenizer:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    rows = list(csv.DictReader(open(a.csv)))
    dur = durations(a.data)
    print(f"{len(rows)} rows, {sum(r['key'] in dur for r in rows)} with a duration")
    for col in a.cols:
        for mode in ("voiced", "silent"):
            analyse(rows, col, mode, dur, sp)


if __name__ == "__main__":
    main()
