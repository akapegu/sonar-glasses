"""Tokenizer wrapper, greedy CTC decoding and WER (README sections 5 and 9).

Class layout: index 0 = CTC blank, index k = SentencePiece id k-1 (so 1,000 pieces -> 1,001 classes).
"""
from pathlib import Path

import jiwer
import sentencepiece as spm
import torch

from common import norm_text


class Tokenizer:
    def __init__(self, model_path):
        self.path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.path)
        self.n_classes = self.sp.get_piece_size() + 1  # + blank

    def encode(self, text: str):
        return [i + 1 for i in self.sp.encode(norm_text(text))]

    def decode(self, ids):
        return self.sp.decode([i - 1 for i in ids if i > 0])


def train_tokenizer(texts, out_prefix: Path, vocab_size: int = 1000):
    """SentencePiece Unigram on normalised training transcripts only."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    txt = out_prefix.with_suffix(".txt")
    txt.write_text("\n".join(norm_text(t) for t in texts) + "\n")
    spm.SentencePieceTrainer.train(
        input=str(txt), model_prefix=str(out_prefix), model_type="unigram", vocab_size=vocab_size,
        character_coverage=1.0, unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1,
        input_sentence_size=0, shuffle_input_sentence=False, num_threads=4, minloglevel=2)
    return Path(str(out_prefix) + ".model")


def greedy_decode(log_probs: torch.Tensor, out_lengths: torch.Tensor, tok: Tokenizer):
    """log_probs (B, T', C): argmax per step, merge repeats, drop blanks, detokenise."""
    best = log_probs.argmax(-1).cpu()
    hyps = []
    for b in range(best.shape[0]):
        seq = best[b, : int(out_lengths[b])].tolist()
        ids, prev = [], 0
        for s in seq:
            if s != prev and s != 0:
                ids.append(s)
            prev = s
        hyps.append(tok.decode(ids))
    return hyps


def wer(refs, hyps):
    """Corpus WER with identical normalisation on both sides."""
    refs = [norm_text(r) for r in refs]
    hyps = [norm_text(h) for h in hyps]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return float("nan")
    return jiwer.wer([r for r, _ in pairs], [h if h else "<empty>" for _, h in pairs])
