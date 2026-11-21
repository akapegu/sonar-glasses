#!/usr/bin/env python3
"""Phase B: build the training cache from extracted SoniSpeech sessions.

For every session (one worker process each, run in parallel):
  1. official calculate_echo_profiles() on audio.wav (unchanged), capturing start_pos
  2. differential profile |p_t| - |p_(t-1)|, 4 paths [s0m0, s0m1, s1m0, s1m1], bins 226:306
  3. one sample per transcript utterance:
       first = floor((start*100000 - start_pos) / 500), last = ceil((end*100000 - start_pos) / 500)
  4. screen: drop utterances on eda/drop_list.csv or with a rail-to-rail jump (> 30000)
  5. save float16 (4, 80, T) as <out>/<mode>_<part>/session_NNN/<utt_id>.npy
     plus <out>/<mode>_<part>/session_NNN/session.json (all rows, kept and dropped)
Then merge every session.json into <out>/manifest.csv.

No normalisation is applied here (statistics come from the train split, at load time).
Resumable: a session with a session.json is skipped unless --overwrite.

Usage (from the repo root):
  python scripts/preprocess.py --data data/SoniSpeech_02_of_09 --out cache/v1
  python scripts/preprocess.py --data data/SoniSpeech_0*_of_09 --out cache/v1 --workers 3
  python scripts/preprocess.py --out cache/v1 --merge-only
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import (CROP, FPS, FRAME, FS, JUMP, PATHS, fix_text, is_faulty,  # noqa: E402
                    norm_text, split_labels)
import official  # noqa: E402

VERSION = 1
FIELDS = ["path", "split_name", "mode", "part", "session", "utt_id", "dev_split", "final_split",
          "faulty_session", "dropped", "n_frames", "start_s", "end_s", "start_pos",
          "text_raw", "text"]


def find_sessions(data_roots):
    out = []
    for root in data_roots:
        for d in sorted(Path(root).glob("SoniSpeech_*_*/session_*")):
            split_name = d.parent.name.replace("SoniSpeech_", "")  # e.g. voiced_train
            mode, part = split_name.split("_")
            if (d / "audio.wav").exists() and (d / "transcript.json").exists():
                out.append(dict(dir=str(d), split_name=split_name, mode=mode, part=part,
                                num=int(d.name.split("_")[1])))
    return out


def load_drop_list(path):
    drops = {}
    if path and Path(path).exists():
        for r in csv.DictReader(open(path)):
            drops[(r["split"], int(r["session"]), r["id"])] = r["reason"]
    return drops


def atomic_save_npy(path: Path, arr):
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def process_session(job):
    s, out, code_dir, drops, overwrite, crop = job
    sdir = Path(s["dir"])
    rel_dir = Path(s["split_name"]) / f"session_{s['num']:03d}"
    odir = Path(out) / rel_dir
    done = odir / "session.json"
    if done.exists() and not overwrite:
        return dict(name=str(rel_dir), skipped=True)
    odir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    profiles, start_pos = official.echo_profiles(Path(code_dir), sdir / "audio.wav")
    d = official.diff_crop(profiles, crop)      # (4, F, n_bins) float32
    n_frames = d.shape[1]
    del profiles

    fs, audio = wavfile.read(sdir / "audio.wav", mmap=True)
    assert fs == FS and audio.ndim == 2 and audio.shape[1] == 2, (fs, audio.shape)
    transcript = json.load(open(sdir / "transcript.json"))
    dev, final = split_labels(s["mode"], s["part"], s["num"])
    faulty = is_faulty(s["mode"], s["part"], s["num"])

    rows, kept = [], 0
    for uid, u in transcript.items():
        f0 = max(0, math.floor((u["start"] * FS - start_pos) / FRAME))
        f1 = min(n_frames, math.ceil((u["end"] * FS - start_pos) / FRAME))
        seg = np.asarray(audio[int(round(u["start"] * FS)): int(round(u["end"] * FS))]).astype(np.int32)
        jumps = (np.abs(np.diff(seg, axis=0)) > JUMP).sum(0) if len(seg) > 1 else np.zeros(2, int)
        reason = drops.get((s["split_name"], s["num"], uid), "")
        if not reason and jumps.any():
            reason = "jump_mic" + "".join(str(c) for c in range(2) if jumps[c])
        if not reason and f1 - f0 < 16:
            reason = "too_short"
        text_raw = fix_text(u["transcript"])
        rel = str(rel_dir / f"{uid}.npy")
        if not reason:
            atomic_save_npy(Path(out) / rel, d[:, f0:f1, :].transpose(0, 2, 1).astype(np.float16))
            kept += 1
        rows.append(dict(path=rel if not reason else "", split_name=s["split_name"], mode=s["mode"],
                         part=s["part"], session=s["num"], utt_id=uid, dev_split=dev, final_split=final,
                         faulty_session=int(faulty), dropped=reason, n_frames=f1 - f0,
                         start_s=u["start"], end_s=u["end"], start_pos=start_pos,
                         text_raw=text_raw, text=norm_text(text_raw)))

    meta = dict(version=VERSION, crop=list(crop), paths=PATHS, fps=FPS, start_pos=start_pos,
                n_frames=n_frames, seconds=round(time.time() - t0, 1), rows=rows)
    tmp = done.with_suffix(".json.tmp")
    json.dump(meta, open(tmp, "w"))
    os.replace(tmp, done)  # written last: its presence means the session is complete
    return dict(name=str(rel_dir), skipped=False, kept=kept, dropped=len(rows) - kept,
                seconds=meta["seconds"])


def merge(out: Path):
    rows, crops = [], set()
    for f in sorted(out.glob("*/session_*/session.json")):
        meta = json.load(open(f))
        rows += meta["rows"]
        crops.add(tuple(meta["crop"]))
    assert len(crops) <= 1, f"sessions in {out} were built with different crops: {crops}"
    rows.sort(key=lambda r: (r["split_name"], r["session"], r["utt_id"]))
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    kept = [r for r in rows if not r["dropped"]]
    by = {}
    for r in kept:
        for k in ("dev_split", "final_split"):
            by.setdefault((k, r[k], r["mode"]), 0)
            by[(k, r[k], r["mode"])] += 1
    summary = dict(sessions=len({(r["split_name"], r["session"]) for r in rows}), utterances=len(rows),
                   kept=len(kept), dropped=len(rows) - len(kept),
                   dropped_by_reason={k: sum(1 for r in rows if r["dropped"] == k)
                                      for k in sorted({r["dropped"] for r in rows if r["dropped"]})},
                   counts={f"{a}:{b}:{c}": v for (a, b, c), v in sorted(by.items())},
                   frames_kept=sum(int(r["n_frames"]) for r in kept),
                   crop=list(crops.pop()) if crops else None)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    return summary


def container_limits():
    """(ram_gb, cpus) visible to this process. Containers (e.g. RunPod) report the whole
    host via sysconf/os.cpu_count, so prefer the cgroup limits when they exist."""
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        ram_gb = 16.0
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        m = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if m != "max":
            ram_gb = min(ram_gb, int(m) / 1e9)
    except (OSError, ValueError):
        pass
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return ram_gb, cpus


def default_workers():
    ram_gb, cpus = container_limits()
    return max(1, min(cpus, int((ram_gb - 4) // 5)))  # ~5 GB peak per session


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="*", default=[], help="extracted zip roots (contain SoniSpeech_*_* folders)")
    ap.add_argument("--out", required=True, help="cache directory")
    ap.add_argument("--code", help="official echo_profiles/ dir (default: found next to the data)")
    ap.add_argument("--drop-list", default=str(ROOT / "eda" / "drop_list.csv"))
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--only", nargs="*", help="limit to sessions like voiced_train/session_001")
    ap.add_argument("--crop", nargs=2, type=int, default=list(CROP), metavar=("FIRST", "END"),
                    help=f"range bins to keep, end exclusive (default {CROP[0]} {CROP[1]})")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.merge_only:
        sess = find_sessions(args.data)
        if args.only:
            want = set(args.only)
            sess = [s for s in sess if f"{s['split_name']}/session_{s['num']:03d}" in want]
        if not sess:
            sys.exit("no sessions found under --data")
        code_dir = Path(args.code) if args.code else official.find_code_dir(*args.data, ROOT / "data")
        drops = load_drop_list(args.drop_list)
        print(f"{len(sess)} sessions, {args.workers} workers, crop {args.crop}, code {code_dir}, "
              f"{len(drops)} drop-list entries", flush=True)
        jobs = [(s, str(out), str(code_dir), drops, args.overwrite, tuple(args.crop)) for s in sess]
        t0, n = time.time(), 0
        with Pool(args.workers, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(process_session, jobs):
                n += 1
                msg = "skipped (done)" if r["skipped"] else f"kept {r['kept']}, dropped {r['dropped']}, {r['seconds']} s"
                print(f"[{n}/{len(jobs)}] {r['name']}: {msg}  (elapsed {time.time() - t0:.0f} s)", flush=True)

    summary = merge(out)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
