#!/usr/bin/env python3
"""Whole-corpus checks after all zips are extracted and scripts/eda.py has run per zip.

Writes:
    eda/eda_full_metrics.json   completeness, corruption, overlap, transcript differences, stats
    eda/eda_findings_full.md    written summary
    eda/drop_list.csv           utterances Phase B must exclude from train/validation (with reason)

Usage (from the project root):
    .venv/bin/python scripts/corpus_check.py
"""
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = "data/SoniSpeech_0*_of_09"
EXPECTED = {"voiced_train": 160, "silent_train": 160, "voiced_eval": 20, "silent_eval": 20}
FAULTY = {"voiced_train": set(range(150, 161)), "silent_train": set(range(141, 161))}
DEV_VAL = range(132, 141)
TEXT_FIXES = {"I appreciate thatDad...": "I appreciate that Dad..."}  # silent_train 078, train_03881


def norm(t):
    t = t.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", t)
    return " ".join(t.split())


def main():
    out = Path("eda")
    # ---- completeness
    seen = defaultdict(list)
    for p in sorted(glob.glob(f"{DATA}/SoniSpeech_*/session_*")):
        sp = p.split("/")[2].replace("SoniSpeech_", "")
        seen[(sp, int(p.split("/")[3][8:]))].append(p.split("/")[1])
    expected = {(sp, i) for sp, n in EXPECTED.items() for i in range(1, n + 1)}
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    dupes = sorted(k for k, v in seen.items() if len(v) > 1)

    # ---- transcripts
    utts = []  # (split, session, id, text)
    for p in sorted(glob.glob(f"{DATA}/SoniSpeech_*/session_*/transcript.json")):
        sp = p.split("/")[2].replace("SoniSpeech_", "")
        n = int(p.split("/")[3][8:])
        for k, v in json.load(open(p)).items():
            utts.append((sp, n, k, v["transcript"], v["end"] - v["start"]))
    by_text = defaultdict(list)
    for u in utts:
        by_text[norm(u[3])].append(u)
    train_txt = {t for t, l in by_text.items() if any(u[0].endswith("train") for u in l)}
    eval_txt = {t for t, l in by_text.items() if any(u[0].endswith("eval") for u in l)}
    overlap = sorted(train_txt & eval_txt)
    cross_session = [t for t, l in by_text.items()
                     if any(len({u[1] for u in l if u[0].endswith(pt)}) > 1 for pt in ("train", "eval"))]

    pair = defaultdict(dict)
    for sp, n, k, t, _ in utts:
        pair[(sp.split("_")[1], n, k)][sp.split("_")[0]] = t
    diffs = [dict(part=pt, session=n, id=k, voiced=v["voiced"], silent=v["silent"])
             for (pt, n, k), v in sorted(pair.items()) if len(v) == 2 and v["voiced"] != v["silent"]]
    fixes = [dict(id=u[2], split=u[0], session=u[1], text=u[3]) for u in utts
             if any(bad in u[3] for bad in TEXT_FIXES)]

    # ---- corruption (from per-zip EDA metrics)
    corrupt = []
    for f in sorted(glob.glob("eda/eda_zip0*_metrics.json")):
        m = json.load(open(f))
        for name, ch, k, jumps in m["corrupt_channels"]:
            for uid in m["corrupt_ids"][name][ch]:
                corrupt.append(dict(session=name, mic=ch, id=uid, session_jumps=jumps))
    per_utt = defaultdict(set)
    for c in corrupt:
        per_utt[(c["session"], c["id"])].add(c["mic"])

    # ---- drop list for Phase B
    drops = []
    for (sess, uid), mics in sorted(per_utt.items()):
        sp, s = sess.split("/")
        reason = "transient_both_mics" if mics == {0, 1} else f"corrupt_mic{''.join(map(str, sorted(mics)))}"
        drops.append(dict(split=sp, session=int(s[8:]), id=uid, reason=reason))
    for t in overlap:
        for u in by_text[t]:
            if u[0].endswith("train"):
                drops.append(dict(split=u[0], session=u[1], id=u[2], reason="sentence_in_test"))
    with open(out / "drop_list.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "session", "id", "reason"])
        w.writeheader()
        w.writerows(drops)

    # ---- stats
    stats = {}
    for mode in ["voiced", "silent"]:
        d = np.array([u[4] for u in utts if u[0].startswith(mode)])
        stats[mode] = dict(utterances=len(d), mean_dur=float(d.mean()), hours=float(d.sum() / 3600))
    starts = [w["best_start"] for f in sorted(glob.glob("eda/eda_zip0*_metrics.json"))
              for w in json.load(open(f))["window_info"].values()]
    faulty_ok = all(not json.load(open(f))["faulty_mismatch"] for f in glob.glob("eda/eda_zip0*_metrics.json"))
    val_drops = [d for d in drops if d["split"].endswith("train") and d["session"] in DEV_VAL]

    metrics = dict(sessions=len(seen), missing=missing, extra=extra, duplicates=dupes,
                   unique_train_sentences=len(train_txt), unique_eval_sentences=len(eval_txt),
                   test_overlap=overlap, sentences_across_session_numbers=len(cross_session),
                   voiced_silent_transcript_diffs=diffs, text_fixes=fixes, corrupt=corrupt,
                   drops=drops, stats=stats, crop_window_starts=starts, faulty_list_confirmed=faulty_ok)
    json.dump(metrics, open(out / "eda_full_metrics.json", "w"), indent=1)

    # ---- markdown
    L = ["# EDA findings (full corpus)", "",
         "Generated by `scripts/corpus_check.py` from all 9 extracted zips and the per-zip EDA metrics.", "",
         "## Completeness", "",
         f"- Sessions: {len(seen)} of {len(expected)}; missing {missing or 'none'}, extra {extra or 'none'}, duplicated {dupes or 'none'}.",
         f"- Utterances: voiced {stats['voiced']['utterances']:,} ({stats['voiced']['hours']:.2f} h, mean {stats['voiced']['mean_dur']:.2f} s), "
         f"silent {stats['silent']['utterances']:,} ({stats['silent']['hours']:.2f} h, mean {stats['silent']['mean_dur']:.2f} s). "
         f"Total {stats['voiced']['hours'] + stats['silent']['hours']:.1f} h (paper: 26.7 h; 5.26 / 5.41 s).",
         f"- Unique sentences: train {len(train_txt):,} (paper 8,000), eval {len(eval_txt):,} (paper 1,000). "
         f"Train is 8,000 + {len(train_txt) - 8000} because {sum(norm(d['voiced']) != norm(d['silent']) for d in diffs)} voiced/silent pairs differ in wording "
         f"({len(diffs)} differ in raw text; the rest only in punctuation).",
         f"- Sentences under more than one session number within a split: {len(cross_session)}. The session number fully determines the sentence set.",
         f"- Official faulty list (31 sessions, mic 1 = copy of mic 0): {'confirmed for every session' if faulty_ok else 'MISMATCH, see per-zip reports'}.",
         "", "## Utterances to drop from train/validation", "",
         f"{len(drops)} utterances, written to `eda/drop_list.csv`. {len(val_drops)} of them are in dev-validation sessions 132–140.", "",
         "| Split | Session | IDs | Reason |", "| --- | --- | --- | --- |"]
    grp = defaultdict(list)
    for d in drops:
        grp[(d["split"], d["session"], d["reason"])].append(d["id"])
    for (sp, s, r), ids in sorted(grp.items(), key=lambda x: -len(x[1])):
        L.append(f"| {sp} | {s:03d} | {len(ids)}: {', '.join(ids) if len(ids) <= 4 else ids[0] + ' … ' + ids[-1]} | {r} |")
    L += ["", "Reasons: `corrupt_mic1` = rail-to-rail int16 jumps on mic 1 (the corruption the dataset README describes, in sessions not on its list); "
          "`transient_both_mics` = a single loud transient on both mics at once (physical, not hardware); "
          "`sentence_in_test` = the sentence also appears in the test set (drop the train copy, never the test copy).", "",
          "## Voiced/silent transcript differences", "",
          "Corrected per mode to what was actually said (paper §3.3: \"5 mismatches … corrected\"). Keep both as they are.", ""]
    for d in diffs:
        L.append(f"- {d['part']} {d['session']:03d} `{d['id']}`: voiced \"{d['voiced']}\" / silent \"{d['silent']}\"")
    L += ["", "## Text fixes for Phase B", ""]
    for f in fixes:
        L.append(f"- `{f['id']}` ({f['split']} {f['session']:03d}): missing space in \"thatDad\" → apply `TEXT_FIXES` before normalisation.")
    L += ["", "## Range crop", "",
          f"- Best 80-bin window start over {len(starts)} clean train sessions (2 per zip): {min(starts)}–{max(starts)}, median {np.median(starts):.0f}. "
          "Frozen crop `226:306` stands.", ""]
    (out / "eda_findings_full.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
