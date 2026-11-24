#!/usr/bin/env python3
"""Step 5b prerequisite: cut the audible speech of every voiced utterance, aligned to its echo frames.

  .venv/bin/python experiments/07_transformer_crosscon/extract_audio.py \
      --data data/SoniSpeech_0*_of_09 --manifest /workspace/cache/v1/manifest.csv --out cache/audio16k

For each kept voiced utterance in the manifest, takes exactly the samples its echo frames cover
(frame f starts at start_pos + f * 500 at 100 kHz, the same arithmetic as scripts/preprocess.py),
averages the two microphones, and resamples 100 kHz -> 16 kHz with a polyphase filter (which
low-passes below 8 kHz, removing the 18-39 kHz chirps). Saves float16 at
<out>/<echo path>.npy, so the audio has exactly n_frames * 80 samples: one echo frame = 80
samples, one HuBERT frame (20 ms) = 4 echo frames, one CTC output step (16x) = 4 HuBERT frames.
"""
import argparse
import csv
import math
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

FS, FRAME, OUT_FS = 100_000, 500, 16_000
SAMPLES_PER_FRAME = FRAME * OUT_FS // FS  # 80


def job(args):
    sdir, rows, out = args
    fs, audio = wavfile.read(Path(sdir) / "audio.wav", mmap=True)
    assert fs == FS and audio.ndim == 2, (fs, audio.shape)
    done = 0
    for r in rows:
        dst = Path(out) / r["path"]
        if dst.exists():
            continue
        f0 = max(0, math.floor((float(r["start_s"]) * FS - int(r["start_pos"])) / FRAME))
        n = int(r["n_frames"])
        a0 = int(r["start_pos"]) + f0 * FRAME
        seg = np.asarray(audio[a0: a0 + n * FRAME], dtype=np.float32).mean(1) / 32768.0
        seg = np.pad(seg, (0, n * FRAME - len(seg)))
        y = resample_poly(seg, 4, 25)[: n * SAMPLES_PER_FRAME]
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp.npy")
        np.save(tmp, y.astype(np.float16))
        tmp.rename(dst)
        done += 1
    return sdir, done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="dataset roots (the extracted SoniSpeech_XX_of_09 folders)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    sessions = {}
    for root in a.data:
        for d in Path(root).glob("SoniSpeech_voiced_*/session_*"):
            sessions[(d.parent.name.replace("SoniSpeech_", ""), int(d.name.split("_")[1]))] = str(d)
    by_session = defaultdict(list)
    for r in csv.DictReader(open(a.manifest)):
        if r["mode"] == "voiced" and r["path"]:
            by_session[(r["split_name"], int(r["session"]))].append(r)
    missing = [k for k in by_session if k not in sessions]
    if missing:
        print(f"warning: {len(missing)} sessions not found locally, e.g. {missing[:3]}")
    jobs = [(sessions[k], rows, a.out) for k, rows in sorted(by_session.items()) if k in sessions]
    print(f"{sum(len(j[1]) for j in jobs)} voiced utterances in {len(jobs)} sessions -> {a.out}", flush=True)
    with Pool(a.procs) as pool:
        for i, (sdir, n) in enumerate(pool.imap_unordered(job, jobs), 1):
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} sessions ({Path(sdir).parent.name}/{Path(sdir).name}: {n} new)", flush=True)


if __name__ == "__main__":
    main()
