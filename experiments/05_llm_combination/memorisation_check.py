#!/usr/bin/env python3
"""Has the LLM memorised the test sentences? (next_steps_spec step 3.5)

Give the model the first K words of a sentence (plain text, no chat template) and let it
continue greedily. Count how often its next M words equal the sentence's next M words.
Compare three sets of equal size:
  test     - SoniSpeech test sentences (sampled from SODA's test split)
  train    - SoniSpeech training sentences (SODA's train split)
  cornell  - Cornell Movie Dialogs lines (a different public source), as a control
If "test" is clearly above "cornell", treat LLM results on test as possibly contaminated.

  /root/venv-llm/bin/python experiments/05_llm_combination/memorisation_check.py \
      --model /root/models/Qwen2.5-7B-Instruct --cache /workspace/cache/v1 \
      --cornell /workspace/lm/external/cornell.txt --out /workspace/lisa/memorisation.json
"""
import argparse
import csv
import json
import random


def load_sets(cache, cornell, n, min_words):
    rows = list(csv.DictReader(open(f"{cache}/manifest.csv")))
    test = sorted({r["text"] for r in rows if r["final_split"] == "test"})
    train = sorted({r["text"] for r in rows if r["final_split"] == "train"})
    corn = [l.strip() for l in open(cornell)]
    rng = random.Random(0)
    out = {}
    for name, pool in [("test", test), ("train", train), ("cornell", corn)]:
        pool = [s for s in pool if len(s.split()) >= min_words]
        out[name] = rng.sample(pool, min(n, len(pool)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--cornell", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=3, help="prompt words")
    ap.add_argument("--m", type=int, default=5, help="continuation words that must match")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    sets = load_sets(a.cache, a.cornell, a.n, a.k + a.m)
    res = {}
    for name, sents in sets.items():
        hits, examples = 0, []
        for s in range(0, len(sents), a.batch):
            chunk = sents[s:s + a.batch]
            prompts = [" ".join(x.split()[:a.k]) for x in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=4 * a.m + 8, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for x, o in zip(chunk, outs):
                want = x.split()[a.k:a.k + a.m]
                got = "".join(c if c.isalpha() or c in "' " else " " for c in o.lower()).split()[:a.m]
                ok = got == want
                hits += ok
                if ok and len(examples) < 10:
                    examples.append(x)
        res[name] = dict(n=len(sents), exact_continuation_rate=hits / len(sents), examples=examples)
        print(f"{name:8s} {hits}/{len(sents)} = {hits / len(sents):.2%} exact {a.m}-word continuations", flush=True)
    res["settings"] = dict(k=a.k, m=a.m, model=a.model)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
