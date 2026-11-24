#!/usr/bin/env python3
"""Combine candidate transcriptions from several models (IMPROVEMENTS.md phase B).

Candidates per utterance: the LM-decoded top-1 of every model in the pool, then the top-10
beams of the first (strongest) model, de-duplicated in that order.

  # B3: ROVER-style word voting over the models' top-1 outputs (no LLM)
  python experiments/05_llm_combination/combine.py rover --decodes A.csv B.csv C.csv --out rover_val
  # B4: off-the-shelf LLM picks/corrects (venv with transformers, GPU)
  python experiments/05_llm_combination/combine.py llm --decodes ... --model /root/models/Qwen2.5-7B-Instruct --out llm_val
  # B5: LoRA fine-tune on validation sessions 132-138, early-stop on 139-140, then apply
  python experiments/05_llm_combination/combine.py finetune --decodes ... --model ... --adapter-out /workspace/lisa/adapter
  python experiments/05_llm_combination/combine.py llm --decodes ... --model ... --adapter /workspace/lisa/adapter --out lisa_test
  # Selection only (no rewriting): rerank the candidates with LLM log-probability + agreement +
  # decoder score + word bonus; weights tuned on one set (dev val), then frozen for another (test)
  python experiments/05_llm_combination/combine.py rerank --decodes ... --model ... --out rerank_val
  python experiments/05_llm_combination/combine.py rerank --decodes ... --model ... --weights rerank_val_weights.json --out rerank_test

--decodes are the *_decode_*.csv files from lm_decode.py decode (same utterance keys in each).
Every command writes <out>.csv (key, mode, ref, hyp) and <out>.json (metrics by mode, from
src/metrics.py: Whisper and our WER, perfect / >50% wrong, position errors, plus the share of
outputs containing a word that is in none of the candidates).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from metrics import report  # noqa: E402

PROMPT = ("Below are candidate transcriptions of one spoken English sentence, produced by different "
          "speech recognizers that read lip and jaw movement. Each contains errors. Write the single "
          "most likely correct sentence. Prefer words that appear in the candidates and change only what "
          "is clearly wrong. It is casual conversational English. Answer with the sentence only, in "
          "lowercase, without punctuation.\n\n")


def load_pool(paths, nbest=10):
    pools = [{r["key"]: r for r in csv.DictReader(open(p))} for p in paths]
    keys = sorted(pools[0])
    utts = []
    for k in keys:
        first = pools[0][k]
        tops = [p[k]["hyp"] for p in pools]
        nb = json.loads(first["nbest"])[:nbest]
        beams = [b[0] for b in nb]
        cands = list(dict.fromkeys([c for c in tops + beams if c.strip()]))
        utts.append(dict(key=k, mode=first["mode"], ref=first["ref"], tops=tops, cands=cands,
                         beam_scores={b[0]: float(b[1]) for b in nb}, session=int(k.split("/")[1])))
    return utts


# ---- B3: ROVER-style voting -------------------------------------------------------------------
def align(pivot, hyp):
    """Levenshtein alignment of two word lists -> list of (pivot_word|None, hyp_word|None)."""
    n, m = len(pivot), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (pivot[i - 1] != hyp[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (pivot[i - 1] != hyp[j - 1]):
            out.append((pivot[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append((pivot[i - 1], None)); i -= 1
        else:
            out.append((None, hyp[j - 1])); j -= 1
    return out[::-1]


def rover(tops):
    """Confusion network on the first hypothesis: each pivot slot (and each gap before it) takes
    the majority word over all hypotheses ("" = deletion); ties keep the pivot's choice."""
    pivot = tops[0].split()
    slots = [[[] for _ in range(len(pivot) + 1)], [[] for _ in range(len(pivot))]]  # gaps, words
    for h in tops:
        gi, wi = 0, 0
        ins = [[] for _ in range(len(pivot) + 1)]
        for p, w in align(pivot, h.split()):
            if p is None:
                ins[wi].append(w)
            else:
                slots[1][wi].append(w or "")
                wi += 1
        for g in range(len(pivot) + 1):
            slots[0][g].append(" ".join(ins[g]))
    out = []
    for g in range(len(pivot) + 1):
        votes = slots[0][g]
        best = max(set(votes), key=lambda v: (votes.count(v), v == ""))
        if best:
            out.append(best)
        if g < len(pivot):
            votes = slots[1][g]
            best = max(set(votes), key=lambda v: (votes.count(v), v == pivot[g]))
            if best:
                out.append(best)
    return " ".join(out)


# ---- shared reporting -----------------------------------------------------------------------
def evaluate(utts, hyps, out, extra=None):
    res = {}
    for m in ["voiced", "silent"]:
        idx = [i for i, u in enumerate(utts) if u["mode"] == m]
        r = report([utts[i]["ref"] for i in idx], [hyps[i] for i in idx])
        cand_words = [set(w for c in utts[i]["cands"] for w in c.split()) for i in idx]
        r["outputs_with_new_word"] = sum(any(w not in cw for w in hyps[i].split())
                                        for i, cw in zip(idx, cand_words)) / len(idx)
        res[m] = r
    if extra:
        res.update(extra)
    json.dump(res, open(f"{out}.json", "w"), indent=1)
    with open(f"{out}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "mode", "ref", "hyp", "candidates"])
        for u, h in zip(utts, hyps):
            w.writerow([u["key"], u["mode"], u["ref"], h, json.dumps(u["cands"])])
    print(f"{Path(out).name}: V {res['voiced']['wer_whisper']:.2%} (ours {res['voiced']['wer_ours']:.2%})  "
          f"S {res['silent']['wer_whisper']:.2%} (ours {res['silent']['wer_ours']:.2%})  new-word outputs "
          f"V {res['voiced']['outputs_with_new_word']:.1%} S {res['silent']['outputs_with_new_word']:.1%}", flush=True)
    return res


def cmd_rover(a):
    utts = load_pool(a.decodes)
    evaluate(utts, [u["tops"][0] for u in utts], a.out + "_top1")
    evaluate(utts, [rover(u["tops"]) for u in utts], a.out)


# ---- B4/B5: LLM -----------------------------------------------------------------------------
def messages(u, target=None):
    m = [{"role": "user", "content": PROMPT + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(u["cands"]))}]
    if target is not None:
        m.append({"role": "assistant", "content": target})
    return m


def clean(text):
    import re
    line = text.strip().split("\n")[0].lower()
    line = re.sub(r"^\s*\d+[.)]\s*", "", line)
    return " ".join(re.sub(r"[^a-z' ]+", " ", line).split())


def load_llm(model_path, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def generate(tok, model, utts, batch=32):
    import torch
    outs = []
    for s in range(0, len(utts), batch):
        prompts = [tok.apply_chat_template(messages(u), tokenize=False, add_generation_prompt=True)
                   for u in utts[s:s + batch]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
        outs += [clean(t) for t in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        print(f"  {min(s + batch, len(utts))}/{len(utts)}", flush=True)
    # an empty answer falls back to the strongest model's top-1
    return [o or u["tops"][0] for o, u in zip(outs, utts)]


def cmd_llm(a):
    utts = load_pool(a.decodes)
    if a.sessions:
        utts = [u for u in utts if u["session"] in set(a.sessions)]
    tok, model = load_llm(a.model, a.adapter)
    evaluate(utts, [u["tops"][0] for u in utts], a.out + "_top1")
    evaluate(utts, generate(tok, model, utts), a.out, extra=dict(model=a.model, adapter=a.adapter))


def cmd_finetune(a):
    import torch
    from peft import LoraConfig, get_peft_model
    utts = load_pool(a.decodes)
    train = [u for u in utts if u["session"] in set(a.train_sessions)]
    held = [u for u in utts if u["session"] in set(a.held_sessions)]
    print(f"fine-tune on {len(train)} utterances, hold out {len(held)}", flush=True)
    tok, model = load_llm(a.model)
    model.gradient_checkpointing_enable()   # 7B + activations do not fit 32 GB otherwise
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"],
                                             task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    def example(u):
        prompt = tok.apply_chat_template(messages(u), tokenize=False, add_generation_prompt=True)
        full = prompt + u["ref"] + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = tok(full, add_special_tokens=False)["input_ids"]
        return ids, [-100] * len(p_ids) + ids[len(p_ids):]

    import random
    rng = random.Random(0)
    base = evaluate(held, generate(tok, model, held), f"{a.adapter_out}_held_epoch0")
    best = (base["voiced"]["wer_whisper"] + base["silent"]["wer_whisper"]) / 2
    model.save_pretrained(a.adapter_out)
    for epoch in range(1, a.epochs + 1):
        model.train()
        rng.shuffle(train)
        tot = 0.0
        for step, s in enumerate(range(0, len(train), a.micro)):
            batch = [example(u) for u in train[s:s + a.micro]]
            L = max(len(i) for i, _ in batch)
            ids = torch.tensor([[tok.pad_token_id] * (L - len(i)) + i for i, _ in batch], device=model.device)
            lab = torch.tensor([[-100] * (L - len(l)) + l for _, l in batch], device=model.device)
            att = (torch.arange(L, device=model.device)[None] >= torch.tensor([L - len(i) for i, _ in batch],
                                                                              device=model.device)[:, None]).long()
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss * a.micro / a.batch).backward()
            if (step + 1) % (a.batch // a.micro) == 0 or s + a.micro >= len(train):
                opt.step()
                opt.zero_grad()
            tot += float(loss) * len(batch) / a.batch
        model.eval()
        r = evaluate(held, generate(tok, model, held), f"{a.adapter_out}_held_epoch{epoch}")
        mean = (r["voiced"]["wer_whisper"] + r["silent"]["wer_whisper"]) / 2
        print(f"epoch {epoch}: train loss {tot / max(1, len(train) // a.batch):.4f}, held-out mean {mean:.2%}", flush=True)
        if mean < best:
            best = mean
            model.save_pretrained(a.adapter_out)
            print(f"  saved adapter (best held-out mean {best:.2%})", flush=True)


# ---- selection-only reranking -----------------------------------------------------------------
RERANK_PREFIX = "The following is a line from a casual English conversation.\n"


def word_distance(a, b):
    return sum(x != y for x, y in align(a.split(), b.split()))


def llm_logprobs(tok, model, texts, batch=64):
    """Sum of log P(token) of each text (plus a closing newline) after a fixed prefix."""
    import torch
    pre = tok(RERANK_PREFIX)["input_ids"]
    seqs = [pre + tok(t + "\n")["input_ids"] for t in texts]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [0.0] * len(seqs)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        L = max(len(seqs[i]) for i in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, :len(seqs[i])] = torch.tensor(seqs[i])
            mask[r, :len(seqs[i])] = 1
        ids, mask = ids.to(model.device), mask.to(model.device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]
        keep = mask[:, 1:].clone()
        keep[:, :len(pre) - 1] = 0  # score only the text, not the prefix
        tot = (lp * keep).sum(1).tolist()
        for r, i in enumerate(idx):
            out[i] = tot[r]
        if (s // batch) % 20 == 0:
            print(f"  scored {min(s + batch, len(order))}/{len(order)}", flush=True)
    return out


def rerank_features(utts, tok, model):
    import jiwer
    from metrics import wnorm
    flat = [c for u in utts for c in u["cands"]]
    lps = llm_logprobs(tok, model, flat)
    k = 0
    for u in utts:
        n = len(u["cands"])
        fill = min(u["beam_scores"].values()) if u["beam_scores"] else 0.0
        R = wnorm(u["ref"])
        u["feat"] = [dict(llm=lps[k + j], agree=-sum(word_distance(c, t) for t in u["tops"]) / len(u["tops"]),
                          dec=u["beam_scores"].get(c, fill), nw=len(c.split()),
                          err=jiwer.process_words(R, wnorm(c)).wer * len(R.split()))
                     for j, c in enumerate(u["cands"])]
        u["ref_len"] = len(R.split())
        k += n


FEATS = ("llm", "agree", "dec", "nw")


def rerank_arrays(utts):
    import numpy as np
    C = max(len(u["cands"]) for u in utts)
    F = np.zeros((len(utts), C, len(FEATS)))
    E = np.zeros((len(utts), C))
    valid = np.zeros((len(utts), C), bool)
    for i, u in enumerate(utts):
        for j, f in enumerate(u["feat"]):
            F[i, j] = [f[k] for k in FEATS]
            E[i, j] = f["err"]
            valid[i, j] = True
    modes = np.array([u["mode"] for u in utts])
    ref_len = np.array([u["ref_len"] for u in utts], float)
    return F, E, valid, modes, ref_len


def rerank_pick(arrs, w):
    import numpy as np
    F, _, valid, _, _ = arrs
    score = F @ np.array([w[k] for k in FEATS], float)
    return np.where(valid, score, -np.inf).argmax(1)


def rerank_wer(arrs, picks):
    import numpy as np
    _, E, _, modes, ref_len = arrs
    err = E[np.arange(len(picks)), picks]
    return float(np.mean([err[modes == m].sum() / ref_len[modes == m].sum() for m in ("voiced", "silent")]))


def cmd_rerank(a):
    import itertools
    utts = load_pool(a.decodes)
    feats = Path(f"{a.out}_features.json")
    if feats.exists():
        saved = json.load(open(feats))
        for u in utts:
            u["feat"], u["ref_len"] = saved[u["key"]]
    else:
        tok, model = load_llm(a.model)
        rerank_features(utts, tok, model)
        json.dump({u["key"]: [u["feat"], u["ref_len"]] for u in utts}, open(feats, "w"))
    arrs = rerank_arrays(utts)
    grid = dict(llm=[0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0], agree=[0, 0.5, 1, 2, 4, 8, 16, 32, 64],
                dec=[0, 0.1, 0.25, 0.5, 1.0], nw=[-1, 0, 1, 2, 3, 4, 6])

    def tune(rows):
        import numpy as np
        sub = tuple(x[rows] for x in arrs)
        log = []
        for vals in itertools.product(*grid.values()):
            w = dict(zip(grid, vals))
            log.append((rerank_wer(sub, rerank_pick(sub, w)), w))
        return {"with_llm": min(log, key=lambda x: x[0])[1],
                "no_llm": min((x for x in log if x[1]["llm"] == 0), key=lambda x: x[0])[1]}

    import numpy as np
    if a.weights:
        weights = json.load(open(a.weights))
    else:  # tune on this set; the no-LLM optimum shows what the LLM adds
        weights = tune(np.arange(len(utts)))
        json.dump(weights, open(f"{a.out}_weights.json", "w"), indent=1)
        print(f"tuned weights: {weights}", flush=True)
    if a.cv_folds:  # tune on some sessions, apply to the others (sentence sets are disjoint by session)
        sessions = sorted({u["session"] for u in utts})
        fold_of = {s_: i % a.cv_folds for i, s_ in enumerate(sessions)}
        folds = np.array([fold_of[u["session"]] for u in utts])
        for name in ("with_llm", "no_llm"):
            picks = np.zeros(len(utts), int)
            for f in range(a.cv_folds):
                w = tune(np.where(folds != f)[0])[name]
                picks[folds == f] = rerank_pick(arrs, w)[folds == f]
                print(f"  cv fold {f} {name}: {w}", flush=True)
            evaluate(utts, [u["cands"][p] for u, p in zip(utts, picks)], f"{a.out}_cv_{name}")
    evaluate(utts, [u["tops"][0] for u in utts], a.out + "_top1")
    evaluate(utts, [rover(u["tops"]) for u in utts], a.out + "_rover")
    for name, w in weights.items():
        picks = rerank_pick(arrs, w)
        evaluate(utts, [u["cands"][p] for u, p in zip(utts, picks)], f"{a.out}_{name}", extra=dict(weights=w))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rover", "llm", "finetune", "rerank"):
        p = sub.add_parser(name)
        p.add_argument("--decodes", nargs="+", required=True, help="first = strongest model")
    sub.choices["rover"].add_argument("--out", required=True)
    p = sub.choices["llm"]
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--sessions", type=int, nargs="*", help="only these sessions (e.g. held-out 139 140)")
    p.add_argument("--out", required=True)
    p = sub.choices["finetune"]
    p.add_argument("--model", required=True)
    p.add_argument("--adapter-out", required=True)
    p.add_argument("--train-sessions", type=int, nargs="+", default=list(range(132, 139)))
    p.add_argument("--held-sessions", type=int, nargs="+", default=[139, 140])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=8, help="effective batch (gradient accumulation)")
    p.add_argument("--micro", type=int, default=4, help="micro-batch per forward pass")
    p = sub.choices["rerank"]
    p.add_argument("--model", required=True)
    p.add_argument("--weights", help="weights JSON from a tuning run (omit to tune on this set)")
    p.add_argument("--cv-folds", type=int, default=0, help="also report session-wise cross-validation")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"rover": cmd_rover, "llm": cmd_llm, "finetune": cmd_finetune, "rerank": cmd_rerank}[a.cmd](a)


if __name__ == "__main__":
    main()
