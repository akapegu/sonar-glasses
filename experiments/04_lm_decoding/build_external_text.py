#!/usr/bin/env python3
"""External conversational text for the LM, normalised like the corpus and leakage-filtered
(IMPROVEMENTS.md phase A2.2-A2.3).

  python experiments/04_lm_decoding/build_external_text.py --cache /workspace/cache/v1 --out /workspace/lm/external \
      --soda /workspace/lm/raw/soda_train.parquet --cornell /workspace/lm/raw/cornell_movie_dialogs_corpus.zip

Sources: SODA *train* split (SoniSpeech's sentences were sampled from SODA, test from SODA
test) and Cornell Movie Dialogs. Each dialogue turn is one line.
Normalisation (as the corpus: lowercase, no punctuation, numbers and currency spelled out):
ASCII only, "$20" -> "twenty dollars", digits -> words (num2words), keep a-z, apostrophes inside
words and spaces; keep turns of 4-40 words.
Leakage filter (required): drop any turn that equals a test or validation sentence (any split
kind, both modes) or shares any 8-word sequence with one. Counts are written to stats.json.
"""
import argparse
import csv
import io
import json
import re
import zipfile
from pathlib import Path

from num2words import num2words

N = 8


def corpus_norm(s: str):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", " ").replace("–", " ").replace("…", " ")
    if not s.isascii():
        return None
    s = re.sub(r"\$(\d+(?:\.\d+)?)", lambda m: f"{m.group(1)} dollars", s)
    s = re.sub(r"(\d+)%", lambda m: f"{m.group(1)} percent", s)
    s = re.sub(r"\d+(?:\.\d+)?", lambda m: " " + num2words(float(m.group()) if "." in m.group() else int(m.group())) + " ", s)
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z' ]+", " ", s)
    s = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", s)   # apostrophes only inside words
    words = s.split()
    return " ".join(words) if 4 <= len(words) <= 40 else None


def ngrams(words, n=N):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def protected(cache):
    with open(Path(cache) / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    sents = {r["text"] for r in rows if r["final_split"] == "test" or r["dev_split"] == "val"}
    grams = set()
    for t in sents:
        grams |= ngrams(t.split())
    return sents, grams


def soda_turns(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["dialogue"])
    for dialogue in table.column("dialogue").to_pylist():
        yield from dialogue


def cornell_turns(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("movie_lines.txt"))
        for line in io.TextIOWrapper(z.open(name), encoding="iso-8859-1"):
            parts = line.rstrip("\n").split(" +++$+++ ")
            if len(parts) == 5:
                yield parts[4]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--soda")
    ap.add_argument("--cornell")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sents, grams = protected(a.cache)
    stats = dict(protected_sentences=len(sents), ngram=N)
    for name, gen in [("soda", soda_turns(a.soda) if a.soda else None),
                      ("cornell", cornell_turns(a.cornell) if a.cornell else None)]:
        if gen is None:
            continue
        s = dict(turns=0, normalised=0, exact_removed=0, ngram_removed=0, kept=0)
        with open(out / f"{name}.txt", "w") as fh:
            for turn in gen:
                s["turns"] += 1
                t = corpus_norm(turn)
                if t is None:
                    continue
                s["normalised"] += 1
                if t in sents:
                    s["exact_removed"] += 1
                    continue
                if ngrams(t.split()) & grams:
                    s["ngram_removed"] += 1
                    continue
                s["kept"] += 1
                fh.write(t + "\n")
        stats[name] = s
        print(name, s, flush=True)
    json.dump(stats, open(out / "stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
