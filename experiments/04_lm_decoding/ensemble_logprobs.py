#!/usr/bin/env python3
"""Average several runs' CTC posteriors per frame (a seed ensemble), for LM decoding.

  python experiments/04_lm_decoding/ensemble_logprobs.py \
      --inputs /workspace/runs/final_vs_final/logprobs_test_last \
               /workspace/runs/final_vs_final_seed1/logprobs_test_last \
               /workspace/runs/final_vs_final_seed2/logprobs_test_last \
      --tokenizer /workspace/runs/final_vs_final/tokenizer.model \
      --out /workspace/runs/ensemble_final_3seeds/logprobs_test_last

All inputs must come from the same tokenizer and the same utterances (equal output lengths).
Combination: log of the mean probability, log((1/N) sum_i p_i). Writes <out>.npz and <out>.csv
in the dump_logprobs.py format, with the ensemble's greedy decode in the "greedy" column, so
lm_decode.py decode works on it unchanged.
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def greedy(lp, sp):
    ids = lp.argmax(-1)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != 0:
            out.append(int(i) - 1)
        prev = i
    return sp.decode(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="logprobs stems (without .npz/.csv)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    zs = [np.load(f"{s}.npz") for s in a.inputs]
    meta = list(csv.DictReader(open(f"{a.inputs[0]}.csv")))
    out = {}
    for m in meta:
        lps = [z[m["key"]].astype(np.float32) for z in zs]
        assert len({x.shape for x in lps}) == 1, f"length mismatch for {m['key']}"
        stack = np.stack(lps)
        mx = stack.max(0)
        ens = mx + np.log(np.exp(stack - mx).mean(0))
        out[m["key"]] = ens.astype(np.float16)
        m["greedy"] = greedy(ens, sp)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{a.out}.npz", **out)
    with open(f"{a.out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(meta[0]))
        w.writeheader()
        w.writerows(meta)
    print(f"ensemble of {len(a.inputs)} runs, {len(out)} utterances -> {a.out}.npz", flush=True)


if __name__ == "__main__":
    main()
