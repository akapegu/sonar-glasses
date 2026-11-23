#!/usr/bin/env python3
"""Phoneme-level evaluation of word outputs: PER, per-phoneme recall, confusions, visemes.

  .venv/bin/python experiments/06_phoneme_analysis/phoneme_eval.py \
      --csv /path/to/logprobs_test_last_decode_final_ext_a0.75_b3.0_w128.csv --cols greedy hyp \
      --out experiments/06_phoneme_analysis/results_seed0

Reference and hypothesis words are converted to ARPABET (CMUdict, with g2p_en's neural
letter-to-sound model for words not in the dictionary, including garbled outputs), stress
marks removed (39 phonemes). Sequences are aligned with Levenshtein distance per sentence.
Reports per mode and column:
  - PER = (S + D + I) / reference phonemes
  - viseme error rate: the same after mapping phonemes to viseme classes (sounds that look
    alike on the face), so confusions within a class are not counted
  - the share of phoneme substitutions that stay within one viseme class
  - per-phoneme recall and the most frequent substitutions
Needs: pip install g2p_en==2.1.0 (downloads CMUdict and a POS tagger via nltk on first use).
"""
import argparse
import csv
import json
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

# Viseme classes for English consonants (place of articulation as seen on the face) and broad
# vowel groups by lip rounding / jaw opening.
VISEMES = {
    "bilabial": "P B M", "labiodental": "F V", "dental": "TH DH",
    "alveolar": "T D N S Z L", "postalveolar": "SH ZH CH JH", "velar": "K G NG",
    "glottal": "HH", "approximant_rw": "R W", "palatal": "Y",
    "vowel_spread": "IY IH EY EH AE", "vowel_open": "AA AH AY AW", "vowel_round": "AO OW OY UH UW ER",
}
P2V = {p: v for v, ps in VISEMES.items() for p in ps.split()}
VOICING = [("P", "B"), ("T", "D"), ("K", "G"), ("F", "V"), ("S", "Z"), ("TH", "DH"), ("SH", "ZH"), ("CH", "JH")]


class Phonemizer:
    def __init__(self):
        import nltk
        for p in ["averaged_perceptron_tagger_eng", "cmudict"]:
            nltk.download(p, quiet=True)
        from g2p_en import G2p
        self.g2p = G2p()
        self.cache = {}

    def word(self, w):
        if w not in self.cache:
            self.cache[w] = [p.rstrip("012") for p in self.g2p(w) if p.strip() and p[0].isalpha()]
        return self.cache[w]

    def sentence(self, s):
        return [p for w in s.lower().split() for p in self.word(w)]


def align(ref, hyp):
    """Levenshtein alignment -> list of (ref_item|None, hyp_item|None)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append((ref[i - 1], None)); i -= 1
        else:
            out.append((None, hyp[j - 1])); j -= 1
    return out[::-1]


def evaluate(pairs, ph):
    S = D = I = N = 0
    vS = vD = vI = 0
    within = voicing = 0
    recall = defaultdict(lambda: [0, 0])
    subs = Counter()
    for ref, hyp in pairs:
        r, h = ph.sentence(ref), ph.sentence(hyp)
        N += len(r)
        for a, b in align(r, h):
            if a is not None:
                recall[a][1] += 1
            if a is not None and b is not None:
                if a == b:
                    recall[a][0] += 1
                else:
                    S += 1
                    subs[(a, b)] += 1
                    within += P2V.get(a) == P2V.get(b)
                    voicing += (a, b) in VOICING or (b, a) in VOICING
            elif a is not None:
                D += 1
            else:
                I += 1
        vr = [P2V.get(p, p) for p in r]
        vh = [P2V.get(p, p) for p in h]
        for a, b in align(vr, vh):
            vS += a is not None and b is not None and a != b
            vD += b is None
            vI += a is None
    return dict(phonemes=N, per=(S + D + I) / N, sub=S / N, dele=D / N, ins=I / N,
                viseme_error_rate=(vS + vD + vI) / N,
                subs_within_viseme=within / max(S, 1), subs_voicing_only=voicing / max(S, 1),
                recall={p: c / t for p, (c, t) in sorted(recall.items()) if t},
                counts={p: t for p, (c, t) in sorted(recall.items())},
                top_subs=[(f"{a}->{b}", n) for (a, b), n in subs.most_common(25)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="a decode CSV with columns mode, ref and the output columns")
    ap.add_argument("--cols", nargs="+", default=["greedy", "hyp"])
    ap.add_argument("--out", required=True, help="output stem (.json)")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    ph = Phonemizer()
    res = {}
    for col in a.cols:
        for mode in ["voiced", "silent"]:
            pairs = [(r["ref"], r[col]) for r in rows if r["mode"] == mode]
            r = evaluate(pairs, ph)
            res[f"{col}/{mode}"] = r
            print(f"{col:7s} {mode:6s} PER {r['per']:.1%} (S {r['sub']:.1%} D {r['dele']:.1%} I {r['ins']:.1%})  "
                  f"viseme ER {r['viseme_error_rate']:.1%}  subs within viseme {r['subs_within_viseme']:.0%}  "
                  f"voicing-only subs {r['subs_voicing_only']:.0%}", flush=True)
    json.dump(res, open(f"{a.out}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
