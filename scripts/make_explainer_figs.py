#!/usr/bin/env python3
"""Render the real-data figures used by explainer/data_and_preprocessing.html.

Needs the cached echo-profile runs from scripts/eda.py (cache/eda/zip02/...).
Usage (from the project root):  .venv/bin/python scripts/make_explainer_figs.py
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal
from scipy.io import wavfile

FS, FRAME, FPS = 100_000, 500, 200
LO = 150  # cached arrays hold range bins 150:400
CROP = (226, 306)
OUT = Path("explainer/img")
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("data/SoniSpeech_02_of_09")
VOICED = ("voiced_train", "session_076")
SILENT = ("silent_train", "session_086")
PATHS = ["s0m0", "s0m1", "s1m0", "s1m1"]

INK, INK2, GRID, PLATE = "#13202b", "#4b5a66", "#dde3e7", "#fbfcfc"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fbfcfc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#eef0f1", "#e34948", "#9e2b2b"])
plt.rcParams.update({"figure.facecolor": PLATE, "axes.facecolor": PLATE, "savefig.facecolor": PLATE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
                     "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def cache(split_sess):
    d = Path("cache/eda/zip02") / f"{split_sess[0]}__{split_sess[1]}"
    st = json.load(open(d / "stats.json"))
    return (np.load(d / "absprof_crop.npy", mmap_mode="r"), np.load(d / "diff_crop.npy", mmap_mode="r"), st["start_pos"])


def frame(t, sp):
    return math.floor((t * FS - sp) / FRAME)


def transcript(split_sess):
    return json.load(open(DATA / f"SoniSpeech_{split_sess[0]}" / split_sess[1] / "transcript.json"))


# 1. chirp train: 25 ms of raw audio
fs, a = wavfile.read(DATA / f"SoniSpeech_{VOICED[0]}" / VOICED[1] / "audio.wav", mmap=True)
t0 = 13.000
x = np.asarray(a[int(t0 * FS): int((t0 + 0.025) * FS), 0], dtype=np.float32) / 32768
f, t, S = signal.spectrogram(x, fs=FS, nperseg=96, noverlap=88, nfft=512)
fig, ax = plt.subplots(figsize=(9, 2.8))
db = 10 * np.log10(S + 1e-14)
ax.pcolormesh(t * 1000, f / 1000, db, cmap=SEQ, vmin=db.max() - 45, vmax=db.max(), shading="auto")
for k in range(6):
    ax.axvline(k * 5, color=INK2, lw=0.6, ls=":")
ax.set_ylim(10, 45)
ax.set_xlabel("time (ms)")
ax.set_ylabel("kHz")
ax.set_title("25 ms of real audio (voiced_train 076, mic 0): two chirps repeat every 5 ms")
save(fig, "fig1_chirps.png")

# 2+3. echo profile and differential profile over 11 s, path s0m0
P, D, sp = cache(VOICED)
tr = transcript(VOICED)
ta, tb = 21.0, 32.0
fa, fb = frame(ta, sp), frame(tb, sp)
for name, arr, cmap, title, sym in [
        ("fig2_profile.png", P, SEQ, "Echo profile |p| (path s0m0): each column is one 5 ms frame, each row one range bin", False),
        ("fig3_diff.png", D, DIV, "Differential profile |p_t| − |p_t−1| (same window): only what moved survives", True)]:
    seg = np.asarray(arr[0, fa:fb, :]).T  # (bins, frames)
    v = np.percentile(np.abs(seg), 99.5)
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.imshow(seg, aspect="auto", origin="lower", cmap=cmap, vmin=-v if sym else 0, vmax=v,
              extent=[ta, tb, LO, LO + seg.shape[0]], interpolation="nearest")
    for u in tr.values():
        if ta < u["start"] < tb:
            ax.axvline(u["start"], color=INK, lw=0.9, ls="--")
        if ta < u["end"] < tb:
            ax.axvline(u["end"], color=INK, lw=0.9, ls="--")
    ax.axhspan(*CROP, facecolor="none", edgecolor="#0f7c86", lw=1.4)
    ax.text(tb - 0.1, CROP[1] + 4, "crop 226:306", color="#0f7c86", ha="right", fontsize=8.5)
    ax.text(26.9, 380, "gap: audio zeroed", color=INK2, ha="center", fontsize=8.5)
    ax.set_xlabel("time in session (s)  ·  dashed = utterance start/end from transcript.json")
    ax.set_ylabel("range bin")
    ax.set_title(title)
    save(fig, name)


# 4. what the model sees: one utterance, 4 paths, cropped + standardised
def model_input(split_sess, uid, fname, title):
    P, D, sp = cache(split_sess)
    u = transcript(split_sess)[uid]
    f0, f1 = math.floor((u["start"] * FS - sp) / FRAME), math.ceil((u["end"] * FS - sp) / FRAME)
    x = np.asarray(D[:, f0:f1, CROP[0] - LO: CROP[1] - LO]).astype(np.float32)  # (4, T, 80)
    full = np.asarray(D[:, :, CROP[0] - LO: CROP[1] - LO])
    nz = np.abs(full).sum(2) > 0  # frames inside utterances (gaps are exact zeros)
    mu = np.array([full[c][nz[c]].mean() for c in range(4)])
    sd = np.array([full[c][nz[c]].std() for c in range(4)])
    x = (x - mu[:, None, None]) / sd[:, None, None]
    fig, axes = plt.subplots(4, 1, figsize=(9, 5.4), sharex=True)
    for c in range(4):
        axes[c].imshow(x[c].T, aspect="auto", origin="lower", cmap=DIV, vmin=-3, vmax=3,
                       extent=[0, x.shape[1] / FPS, CROP[0], CROP[1]], interpolation="nearest")
        axes[c].set_ylabel(PATHS[c], rotation=0, ha="right", va="center")
        axes[c].set_yticks([CROP[0], CROP[1]])
    axes[0].set_title(f"{title}\n“{u['transcript']}”", fontsize=10)
    axes[-1].set_xlabel(f"seconds  ·  tensor shape (4, 80, {x.shape[1]})  ·  colour = standardised value, ±3σ")
    fig.tight_layout()
    save(fig, fname)
    return x.shape


print(model_input(VOICED, "train_03755", "fig4_input_voiced.png", "Model input, voiced_train 076 / train_03755"))
print(model_input(SILENT, "train_04251", "fig5_input_silent.png", "Model input, silent_train 086 / train_04251"))
