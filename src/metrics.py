"""Evaluation metrics shared by the decoders (IMPROVEMENTS.md section 0).

All alignment metrics work on Whisper-normalised text (the headline scoring) and count
reference-word errors (substitutions + deletions; insertions have no reference position).
"""
import sys
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import norm_text  # noqa: E402
from whisper_normalizers import EnglishTextNormalizer  # noqa: E402

WHISPER = EnglishTextNormalizer()


def wnorm(s: str) -> str:
    return WHISPER(s) or "<empty>"


def onorm(s: str) -> str:
    return norm_text(s) or "<empty>"


def ref_word_errors(ref: str, hyp: str):
    """Per reference word: True if it was substituted or deleted."""
    out = jiwer.process_words(ref, hyp)
    wrong = [False] * len(ref.split())
    for a in out.alignments[0]:
        if a.type in ("substitute", "delete"):
            for i in range(a.ref_start_idx, a.ref_end_idx):
                wrong[i] = True
    return wrong


def report(refs, hyps, oov_words=frozenset()):
    """WER (Whisper and ours), perfect / >50%-wrong sentence rates, OOV recall, WER by position."""
    R = [wnorm(r) for r in refs]
    H = [wnorm(h) for h in hyps]
    res = dict(n=len(refs), wer_whisper=jiwer.wer(R, H),
               wer_ours=jiwer.wer([onorm(r) for r in refs], [onorm(h) for h in hyps]))
    perfect = half = 0
    pos = {"first": [0, 0], "first3": [0, 0], "middle": [0, 0], "last": [0, 0]}
    oov_hit = oov_n = 0
    for r, h in zip(R, H):
        wrong = ref_word_errors(r, h)
        words = r.split()
        n = len(words)
        perfect += jiwer.wer(r, h) == 0
        half += jiwer.wer(r, h) > 0.5
        for i, (w, bad) in enumerate(zip(words, wrong)):
            keys = (["first"] if i == 0 else []) + (["first3"] if i < 3 else []) \
                + (["last"] if i == n - 1 else []) + (["middle"] if 3 <= i < n - 1 else [])
            for k in keys:
                pos[k][0] += bad
                pos[k][1] += 1
            if w in oov_words:
                oov_n += 1
                oov_hit += not bad
    res.update(perfect=perfect / len(R), over_half_wrong=half / len(R),
               oov_recall=oov_hit / oov_n if oov_n else float("nan"), oov_tokens=oov_n,
               **{f"err_{k}": e / max(c, 1) for k, (e, c) in pos.items()})
    return res
