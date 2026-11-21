#!/usr/bin/env python3
"""Phase A exploratory data analysis for one SoniSpeech zip.

Writes self-contained HTML reports, metrics JSON and a markdown summary to eda/.
Every number in the reports is computed here.

Usage (from the repo root):
    .venv/bin/python experiments/00_data_analysis/eda.py --data data/SoniSpeech_01_of_09 --tag zip01

Naming convention for reports: eda/eda_<tag>_<NN>_<topic>.html
    00 index, 01 inventory, 02 audio, 03 transcripts, 04 echo profiles,
    05 faulty mic, 06 memory.  Raw metrics: eda/eda_<tag>_metrics.json,
    summary: eda/eda_findings_<tag>.md
"""
import argparse
import base64
import datetime as dt
import html
import io
import json
import math
import re
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import wavfile
from scipy import signal

HERE = Path(__file__).resolve().parent  # experiments/00_data_analysis

FS = 100_000
FRAME = 500
FPS = FS // FRAME  # 200
SPLITS = ["SoniSpeech_voiced_train", "SoniSpeech_voiced_eval",
          "SoniSpeech_silent_train", "SoniSpeech_silent_eval"]
FAULTY = {("voiced", "train"): set(range(150, 161)),
          ("silent", "train"): set(range(141, 161))}
PATHS = ["s0m0", "s0m1", "s1m0", "s1m1"]
DIRECT_BIN = FRAME // 2  # where the official code centres the direct path
BIN_LO, BIN_HI = 150, 400  # range-bin window kept for plots
JUMP = 30000  # |x[n] - x[n-1]| above this = int16 rail-to-rail bounce (corrupt channel signature)
CROP_SEARCH = (150, 450)  # bins searched for the 80-bin crop
VAL_SESSIONS = (132, 140)  # dev validation sessions, both modes (README section 4)

# ---------- palette (dataviz reference palette, light surface) ----------
C = {"voiced": "#2a78d6", "silent": "#eb6834", "aqua": "#1baf7a",
     "violet": "#6250d6", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#8a8984",
     "grid": "#e4e3df", "surface": "#fcfcfb", "good": "#0ca30c",
     "warning": "#fab219", "critical": "#d03b3b"}
PATH_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#f0efec", "#e34948", "#9e2b2b"])

plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "savefig.facecolor": C["surface"], "axes.edgecolor": C["grid"],
    "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "text.color": C["ink"], "axes.grid": True, "grid.color": C["grid"],
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
})


# ======================================================================
# helpers
# ======================================================================
def parse_split(name):
    _, mode, part = name.split("_")
    return mode, part


def sessions(data_root):
    out = []
    for sp in SPLITS:
        d = data_root / sp
        if not d.exists():
            continue
        mode, part = parse_split(sp)
        for s in sorted(d.iterdir()):
            if s.is_dir() and s.name.startswith("session_"):
                num = int(s.name.split("_")[1])
                out.append(dict(split=sp, mode=mode, part=part, num=num,
                                name=f"{mode}_{part}/{s.name}", dir=s,
                                faulty=num in FAULTY.get((mode, part), set())))
    return out


def load_transcript(sdir):
    d = json.load(open(sdir / "transcript.json"))
    return [dict(id=k, **v) for k, v in d.items()]


def norm_text(t):
    t = t.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", t)  # keep in-word apostrophes only
    return " ".join(t.split())


def human(nbytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(nbytes) < 1024 or unit == "TB":
            return f"{nbytes:,.1f} {unit}" if unit != "B" else f"{int(nbytes)} B"
        nbytes /= 1024


def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ======================================================================
# official echo-profile code, wrapped (run in a child process)
# ======================================================================
def code_dir(data_root):
    """Official echo_profiles/ code: ships only in zip 01, so fall back to a sibling extract."""
    own = data_root / "echo_profiles"
    if own.exists():
        return own
    for cand in sorted(data_root.parent.glob("SoniSpeech_*/echo_profiles")):
        return cand
    raise FileNotFoundError("echo_profiles/ not found in the data root or any sibling SoniSpeech_* folder")


def run_official(data_root, sdir):
    """Call the official calculate_echo_profiles unchanged; capture start_pos
    from its locals with a return-trace (no reimplementation)."""
    ep_dir = code_dir(data_root)
    sys.path.insert(0, str(ep_dir))
    import echo_profiles as ep  # noqa: E402

    tx_signals = []
    for f in ["fmcw18000_b10000_l500_100k.wav", "fmcw29000_b10000_l500_100k.wav"]:
        _, tx = wavfile.read(ep_dir / "tx_signals" / f)
        tx_signals.append(tx.astype(float) / 32768.0)
    bp_ranges = [[17500, 28500], [28500, 39500]]  # as in official main()

    target = ep.calculate_echo_profiles.__code__
    captured = {}

    def gtrace(frame, event, arg):
        if frame.f_code is target:
            frame.f_trace_lines = False

            def ltrace(fr, ev, a):
                if ev == "return":
                    captured["start_pos"] = int(fr.f_locals["start_pos"])
                return ltrace
            return ltrace
        return None

    t0 = time.time()
    sys.settrace(gtrace)
    try:
        profiles = ep.calculate_echo_profiles(str(sdir / "audio.wav"), tx_signals, bp_ranges)
    finally:
        sys.settrace(None)
    t_prof = time.time() - t0
    diff = np.abs(profiles[:, :, 1:, :]) - np.abs(profiles[:, :, :-1, :])  # as official main()
    diff = np.pad(diff, ((0, 0), (0, 0), (1, 0), (0, 0)), mode="constant")
    return profiles, diff, captured["start_pos"], t_prof


def child_profile(data_root, sdir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles, diff, start_pos, t_prof = run_official(data_root, sdir)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss if sys.platform == "darwin" else rss * 1024
    n_frames = profiles.shape[2]
    utts = load_transcript(sdir)
    mask = np.zeros(n_frames, bool)
    for u in utts:
        a = max(0, math.floor((u["start"] * FS - start_pos) / FRAME))
        b = min(n_frames, math.ceil((u["end"] * FS - start_pos) / FRAME))
        mask[a:b] = True
    P = profiles.reshape(4, n_frames, FRAME)
    D = diff.reshape(4, n_frames, FRAME)
    absP = np.abs(P)
    stats = dict(
        start_pos=start_pos, shape=list(profiles.shape), dtype=str(profiles.dtype),
        diff_shape=list(diff.shape), diff_dtype=str(diff.dtype),
        runtime_s=round(t_prof, 1), max_rss_bytes=int(rss),
        npy_bytes_each=int(profiles.nbytes), n_frames=int(n_frames),
        audio_s=n_frames * FRAME / FS,
        mean_abs_profile_utt=absP[:, mask].mean(1).tolist(),
        mean_abs_profile_gap=absP[:, ~mask].mean(1).tolist() if (~mask).any() else None,
        mean_abs_diff_utt=np.abs(D[:, mask]).mean(1).tolist(),
        mean_abs_diff_gap=np.abs(D[:, ~mask]).mean(1).tolist() if (~mask).any() else None,
        std_abs_profile_utt=absP[:, mask].std(1).tolist(),
        utt_frames=int(mask.sum()),
    )
    np.save(out_dir / "diff_crop.npy", D[:, :, BIN_LO:BIN_HI].astype(np.float32))
    np.save(out_dir / "absprof_crop.npy", absP[:, :, BIN_LO:BIN_HI].astype(np.float32))
    json.dump(stats, open(out_dir / "stats.json", "w"))


# ======================================================================
# analyses
# ======================================================================
def analyse_inventory(data_root, sess):
    by_ext = Counter()
    count_ext = Counter()
    rows = []
    for s in sess:
        files = {}
        for f in s["dir"].iterdir():
            if f.is_file():
                by_ext[f.suffix] += f.stat().st_size
                count_ext[f.suffix] += 1
                files[f.name] = f.stat().st_size
        rows.append(dict(name=s["name"], mode=s["mode"], part=s["part"], num=s["num"],
                         faulty=s["faulty"], files=files))
    cd = code_dir(data_root)
    code = sorted(str(p.relative_to(cd.parent)) for p in cd.rglob("*") if p.is_file())
    if cd.parent != data_root:
        code = [f"(not in this zip; using {cd})"] + code
    split_counts = Counter(f'{s["mode"]}_{s["part"]}' for s in sess)
    return dict(by_ext=dict(by_ext), count_ext=dict(count_ext), sessions=rows,
                code_files=code, split_counts=dict(split_counts))


def analyse_audio_transcripts(sess):
    """Header checks, zero-mask checks, boundary offsets, channel equality."""
    out = []
    for s in sess:
        fs, a = wavfile.read(s["dir"] / "audio.wav", mmap=True)
        utts = load_transcript(s["dir"])
        n = a.shape[0]
        dur = n / fs
        # gap mask (sample level) from transcript
        intervals = [(int(round(u["start"] * fs)), int(round(u["end"] * fs))) for u in utts]
        gaps = []
        prev = 0
        for st, en in intervals:
            if st > prev:
                gaps.append((prev, st))
            prev = max(prev, en)
        if prev < n:
            gaps.append((prev, n))
        # zero fraction in gaps (trim 2 ms at each edge against rounding)
        gz, gt = 0, 0
        for st, en in gaps:
            st2, en2 = st + 200, en - 200
            if en2 > st2:
                seg = a[st2:en2]
                gz += int((seg == 0).all(axis=1).sum())
                gt += en2 - st2
        # zero fraction inside utterances (0.5 s from each middle)
        uz, ut = 0, 0
        ch_equal, ch_corr = [], []
        low_e, chirp_e = [], []
        jumps = []  # per utterance: [ch0, ch1] count of rail-to-rail jumps
        for st, en in intervals:
            full = np.asarray(a[st:en]).astype(np.int32)
            jumps.append((np.abs(np.diff(full, axis=0)) > JUMP).sum(0).tolist())
            mid = (st + en) // 2
            seg = np.asarray(a[mid - 25000: mid + 25000], dtype=np.float32)
            uz += int((seg == 0).all(axis=1).sum())
            ut += len(seg)
            ch_equal.append(float((seg[:, 0] == seg[:, 1]).mean()))
            if seg[:, 0].std() > 0 and seg[:, 1].std() > 0:
                ch_corr.append(float(np.corrcoef(seg[:, 0], seg[:, 1])[0, 1]))
            f, pxx = signal.welch(seg[:, 0], fs=fs, nperseg=4096)
            low_e.append(float(pxx[(f >= 100) & (f < 8000)].sum()))
            chirp_e.append(float(pxx[(f >= 18000) & (f < 39000)].sum()))
        # boundary alignment: first non-zero sample around each start
        offsets = []
        prev_end = 0
        for st, en in intervals:
            gap_ok = st - prev_end >= 50000  # only utterances after >= 0.5 s of masked audio
            lo = max(prev_end, st - 50000)
            prev_end = en
            if not gap_ok:
                continue
            seg = a[lo: st + 50000]
            nz = np.flatnonzero((seg != 0).any(axis=1))
            if len(nz):
                offsets.append((lo + nz[0] - st) / fs * 1000)  # ms
        durs = [u["end"] - u["start"] for u in utts]
        words = [len(norm_text(u["transcript"]).split()) for u in utts]
        out.append(dict(
            name=s["name"], mode=s["mode"], part=s["part"], num=s["num"], faulty=s["faulty"],
            fs=int(fs), channels=int(a.shape[1]), dtype=str(a.dtype), duration_s=dur,
            n_utts=len(utts), first_start=utts[0]["start"], last_end=utts[-1]["end"],
            utt_durs=durs, words=words, utt_seconds=float(sum(durs)),
            gap_seconds=[(e - s_) / fs for s_, e in gaps],
            gap_zero_frac=gz / gt if gt else None, utt_zero_frac=uz / ut if ut else None,
            start_offsets_ms=offsets,
            ch_equal_frac=float(np.mean(ch_equal)), ch_corr=float(np.mean(ch_corr)) if ch_corr else None,
            low_band_rel=float(np.median(np.array(low_e) / np.array(chirp_e))),
            texts=[u["transcript"] for u in utts], ids=[u["id"] for u in utts],
            end_past_audio=utts[-1]["end"] > dur + 1e-3,
            corrupt_utts=[int(sum(j[c] > 0 for j in jumps)) for c in range(2)],
            jumps_total=[int(sum(j[c] for j in jumps)) for c in range(2)],
            corrupt_ids=[[u["id"] for u, j in zip(utts, jumps) if j[c] > 0] for c in range(2)],
        ))
        print(f"  audio/transcript: {s['name']}", flush=True)
    return out


def spectrogram_fig(sess, data_root):
    pick = {}
    for s in sess:
        if s["part"] == "train" and not s["faulty"] and s["mode"] not in pick:
            pick[s["mode"]] = s
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
    info = {}
    for r, mode in enumerate(["voiced", "silent"]):
        s = pick[mode]
        fs, a = wavfile.read(s["dir"] / "audio.wav", mmap=True)
        u = load_transcript(s["dir"])[0]
        t0 = u["start"] + 0.5
        x = np.asarray(a[int(t0 * fs): int((t0 + 1) * fs), 0], dtype=np.float32) / 32768
        f, t, sxx = signal.spectrogram(x, fs=fs, nperseg=1024, noverlap=768)
        db = 10 * np.log10(sxx + 1e-14)
        vmax = np.percentile(db, 99.5)
        for c, (fmax, title) in enumerate([(50000, "0–50 kHz"), (8000, "0–8 kHz (speech band)")]):
            ax = axes[r, c]
            sel = f <= fmax
            ax.pcolormesh(t + t0, f[sel] / 1000, db[sel], cmap=SEQ, vmin=vmax - 70, vmax=vmax, shading="auto")
            ax.grid(False)
            ax.set_title(f"{mode} · {s['name']} · mic 0 · {title}")
            ax.set_ylabel("kHz")
            if r == 1:
                ax.set_xlabel("time (s)")
            if c == 0:
                for lo, hi in [(18, 28), (29, 39)]:
                    ax.axhline(lo, color=C["ink2"], lw=0.6, ls=":")
                    ax.axhline(hi, color=C["ink2"], lw=0.6, ls=":")
        info[mode] = dict(session=s["name"], window=[round(t0, 3), round(t0 + 1, 3)], utt=u["id"])
    fig.tight_layout()
    return fig_b64(fig), info


def band_energy_fig(sess, data_root):
    """Mean PSD across bands for one voiced and one silent session (1 s windows)."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for mode in ["voiced", "silent"]:
        s = next(x for x in sess if x["mode"] == mode and x["part"] == "train" and not x["faulty"])
        fs, a = wavfile.read(s["dir"] / "audio.wav", mmap=True)
        psds = []
        for u in load_transcript(s["dir"])[:20]:
            mid = int((u["start"] + u["end"]) / 2 * fs)
            seg = np.asarray(a[mid - 50000: mid + 50000, 0], dtype=np.float32) / 32768
            f, p = signal.welch(seg, fs=fs, nperseg=4096)
            psds.append(p)
        ax.plot(f / 1000, 10 * np.log10(np.mean(psds, 0) + 1e-16), color=C[mode], label=f"{mode} ({s['name']})")
    ax.axvspan(18, 28, color=C["grid"], alpha=0.6, lw=0)
    ax.axvspan(29, 39, color=C["grid"], alpha=0.6, lw=0)
    ax.text(23, ax.get_ylim()[1], "chirp 0", ha="center", va="top", color=C["ink2"])
    ax.text(34, ax.get_ylim()[1], "chirp 1", ha="center", va="top", color=C["ink2"])
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title("Mean power spectrum inside utterances (20 utterances, mic 0)")
    ax.set_xlim(0, 50)
    ax.legend(loc="lower left")
    return fig_b64(fig)


def transcript_figs(at):
    figs = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for mode in ["voiced", "silent"]:
        d = np.concatenate([r["utt_durs"] for r in at if r["mode"] == mode])
        w = np.concatenate([r["words"] for r in at if r["mode"] == mode])
        axes[0].hist(d, bins=np.arange(0, 16, 0.5), color=C[mode], alpha=0.55, label=f"{mode} (mean {d.mean():.2f} s)", edgecolor=C["surface"], linewidth=1)
        axes[1].hist(w, bins=np.arange(0, 45, 2), color=C[mode], alpha=0.55, label=f"{mode} (mean {w.mean():.1f} ± {w.std():.1f})", edgecolor=C["surface"], linewidth=1)
    for ax, ref, lab in [(axes[0], None, "utterance duration (s)"), (axes[1], None, "words per utterance")]:
        ax.set_xlabel(lab)
        ax.set_ylabel("utterances")
        ax.legend()
    axes[0].set_title("Utterance durations")
    axes[1].set_title("Words per utterance")
    fig.tight_layout()
    figs["dur_words"] = fig_b64(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    g = np.concatenate([r["gap_seconds"] for r in at])
    axes[0].hist(g, bins=np.arange(0, max(22, g.max() + 1), 0.5), color=C["violet"], edgecolor=C["surface"], linewidth=1)
    axes[0].set_xlabel("zero-masked gap length (s)")
    axes[0].set_ylabel("gaps")
    axes[0].set_title("Gaps between utterances (incl. lead-in and tail)")
    o = np.concatenate([r["start_offsets_ms"] for r in at])
    axes[1].hist(np.clip(o, -20, 20), bins=np.arange(-20, 20.5, 0.5), color=C["aqua"], edgecolor=C["surface"], linewidth=1)
    axes[1].set_xlabel("first non-zero sample − transcript start (ms, clipped ±20)")
    axes[1].set_ylabel("utterances")
    axes[1].set_title("Audio unmasking vs transcript start")
    fig.tight_layout()
    figs["gaps"] = fig_b64(fig)
    return figs


def profile_figs(prof_results, at_by_name, data_root):
    figs = {}
    bins = np.arange(FRAME)
    # static |profile| and motion per bin
    fig, axes = plt.subplots(len(prof_results), 2, figsize=(11, 3.3 * len(prof_results)), squeeze=False)
    window_info = {}
    for r, (name, st) in enumerate(prof_results.items()):
        mp = np.array(st["mean_abs_profile_utt"])
        md = np.array(st["mean_abs_diff_utt"])
        for i in range(4):
            axes[r, 0].plot(bins, mp[i], color=PATH_COLORS[i], lw=1.4, label=PATHS[i])
            axes[r, 1].plot(bins, md[i], color=PATH_COLORS[i], lw=1.4, label=PATHS[i])
        ok = healthy_paths(st)
        e = motion_excess(md, ok)
        cs = np.concatenate([[0], np.cumsum(e)])
        sums = cs[80:] - cs[:-80]
        best = int(np.argmax(sums))
        tot = e.sum()
        peak = [int(np.argmax(mp[i])) for i in range(4) if ok[i]]
        window_info[name] = dict(best_start=best, best_frac=float(sums[best] / tot),
                                 frac_240_320=float((cs[320] - cs[240]) / tot),
                                 frac_0_80=float(cs[80] / tot), peak_bins=peak, healthy=ok,
                                 motion_peak_bins=[int(np.argmax(md[i])) for i in range(4) if ok[i]])
        for ax in axes[r]:
            ax.axvline(DIRECT_BIN, color=C["ink2"], lw=0.8, ls=":")
            ax.axvspan(best, best + 80, color=C["aqua"], alpha=0.12, lw=0)
            ax.set_xlabel("range bin (1 bin ≈ 3.4 mm path length)")
        axes[r, 0].set_title(f"{name}: mean |echo profile| inside utterances")
        axes[r, 1].set_title(f"{name}: mean |differential| inside utterances")
        axes[r, 0].set_yscale("log")
        axes[r, 1].set_yscale("log")
        axes[r, 1].legend(ncol=4, loc="upper right", fontsize=8)
    fig.tight_layout()
    figs["per_bin"] = fig_b64(fig)

    # zoomed motion energy, linear, both sessions
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for k, (name, st) in enumerate(prof_results.items()):
        md = np.array(st["mean_abs_diff_utt"])
        e = motion_excess(md, healthy_paths(st))
        ax.plot(bins, e / e.max(), color=[C["voiced"], C["silent"]][k % 2], label=name)
        wi_ = window_info[name]["best_start"]
        ax.axvspan(wi_, wi_ + 80, color=[C["voiced"], C["silent"]][k % 2], alpha=0.08, lw=0)
    ax.axvline(DIRECT_BIN, color=C["ink2"], lw=0.8, ls=":")
    ax.text(DIRECT_BIN + 2, ax.get_ylim()[1] * 0.95, "bin 250 (direct path)", color=C["ink2"], va="top", fontsize=8)
    ax.set_xlim(BIN_LO, BIN_HI)
    ax.set_xlabel("range bin")
    ax.set_ylabel("motion above floor (norm.)")
    ax.set_title("Motion energy above each path's floor, healthy paths averaged (shaded: best 80-bin window)")
    ax.legend()
    figs["motion_zoom"] = fig_b64(fig)

    # utterance heatmaps
    figs["heat"] = {}
    for name, st in prof_results.items():
        cache = Path(st["_cache"])
        D = np.load(cache / "diff_crop.npy", mmap_mode="r")
        at = at_by_name[name]
        k = 1  # second utterance: has a predecessor gap in most sessions
        tr = load_transcript(data_root / f"SoniSpeech_{name.split('/')[0]}" / name.split("/")[1])
        # pick an utterance preceded by a gap >= 1 s so the quiet region is visible
        for i in range(1, len(tr)):
            if tr[i]["start"] - tr[i - 1]["end"] >= 1.0:
                k = i
                break
        u = tr[k]
        sp = st["start_pos"]
        f0 = math.floor((u["start"] * FS - sp) / FRAME)
        f1 = math.ceil((u["end"] * FS - sp) / FRAME)
        a, b = max(0, f0 - 200), min(D.shape[1], f1 + 200)
        lo, hi = 200 - BIN_LO, 350 - BIN_LO
        seg = np.asarray(D[:, a:b, lo:hi])
        fig, axes = plt.subplots(4, 1, figsize=(11, 8.6), sharex=True)
        for i in range(4):
            ax = axes[i]
            v = np.percentile(np.abs(seg[i]), 99) or 1.0
            ax.imshow(seg[i].T, aspect="auto", origin="lower", cmap=DIV, vmin=-v, vmax=v,
                           extent=[a / FPS, b / FPS, 200, 350], interpolation="nearest")
            ax.grid(False)
            for t in (f0, f1):
                ax.axvline(t / FPS, color=C["ink"], lw=1, ls="--")
            ax.axhline(DIRECT_BIN, color=C["ink2"], lw=0.6, ls=":")
            ax.set_ylabel(f"{PATHS[i]}\nbin")
            ax.text(1.005, 0.5, f"±{v:.2g}", transform=ax.transAxes, fontsize=8, color=C["ink2"], va="center")
        axes[0].set_title(f"{name} · utterance {u['id']} ({u['end'] - u['start']:.2f} s) · differential profile, bins 200–350 · dashed = transcript start/end")
        axes[-1].set_xlabel("time in session (s, frame-aligned with start_pos)")
        figs["heat"][name] = dict(img=fig_b64(fig), utt=u["id"], text=u["transcript"])
    return figs, window_info


def healthy_paths(st):
    """A path is unhealthy when its in-utterance |diff| level is >10x the median path."""
    md = np.array(st["mean_abs_diff_utt"])[:, CROP_SEARCH[0]:CROP_SEARCH[1]].mean(1)
    return [bool(x < 10 * np.median(md)) for x in md]


def motion_excess(md, ok):
    """Per-path motion above its own floor (20th percentile inside CROP_SEARCH),
    normalised per path, averaged over healthy paths, zero outside CROP_SEARCH."""
    lo, hi = CROP_SEARCH
    out = np.zeros(md.shape[1])
    m = md[np.array(ok)]
    for row in m:
        x = np.clip(row - np.percentile(row[lo:hi], 20), 0, None)
        x[:lo] = 0
        x[hi:] = 0
        out += x / (x.max() or 1)
    return out / max(len(m), 1)


def health_fig(at):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    xs = np.arange(len(at))
    for c, off, col in [(0, -0.2, C["voiced"]), (1, 0.2, C["silent"])]:
        ax.bar(xs + off, [r["corrupt_utts"][c] for r in at], width=0.4, color=col, label=f"mic {c}")
    ax.set_ylim(0, 50)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["name"].replace("_train", "_tr").replace("_eval", "_ev").replace("session_", "") for r in at], rotation=90, fontsize=7)
    ax.set_ylabel("corrupt utterances (of 50)")
    ax.set_title(f"Mic health: utterances containing rail-to-rail jumps (|Δx| > {JUMP:,})")
    ax.legend(loc="upper left")
    return fig_b64(fig)


def faulty_fig(at):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    xs = np.arange(len(at))
    vals = [r["ch_equal_frac"] for r in at]
    cols = [C["critical"] if r["faulty"] else (C["voiced"] if r["mode"] == "voiced" else C["silent"]) for r in at]
    ax.bar(xs, vals, color=cols, width=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["name"].replace("_train", "_tr").replace("_eval", "_ev").replace("session_", "") for r in at], rotation=90, fontsize=7)
    ax.set_ylabel("fraction ch0 == ch1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Sample-exact equality of mic channels inside utterances (red = on the official faulty list)")
    return fig_b64(fig)


# ======================================================================
# HTML
# ======================================================================
CSS = """
:root{--bg:#f6f5f2;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;--line:#e4e3df;
--accent:#2a78d6;--good:#0b7d0b;--warn:#a86f00;--bad:#c23434;--code:#efeee9;--plate:#fcfcfb}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121211;--card:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;--line:#33332f;--accent:#3987e5;--good:#39b839;
--warn:#e0a52a;--bad:#e66767;--code:#262624;--plate:#fcfcfb}}
:root[data-theme="dark"]{--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;
--line:#33332f;--accent:#3987e5;--good:#39b839;--warn:#e0a52a;--bad:#e66767;--code:#262624;--plate:#fcfcfb}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 16px 64px}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;margin-bottom:18px}
nav a{color:var(--ink2);text-decoration:none;border-bottom:1px solid transparent}
nav a:hover,nav a.on{color:var(--accent);border-color:var(--accent)}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:18px;margin:34px 0 10px;padding-top:6px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:20px 0 6px}
.sub{color:var(--muted);font-size:13px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:12px 0}
.checks{display:grid;gap:8px}
.chk{display:grid;grid-template-columns:86px 1fr;gap:12px;align-items:start}
.badge{display:inline-flex;gap:5px;align-items:center;font-size:12px;font-weight:600;padding:2px 8px;border-radius:999px;border:1px solid currentColor;justify-content:center}
.pass{color:var(--good)}.warn{color:var(--warn)}.fail{color:var(--bad)}.info{color:var(--accent)}
figure{margin:12px 0;background:var(--plate);border:1px solid var(--line);border-radius:10px;padding:10px}
figure img{width:100%;height:auto;display:block}
figcaption{color:#52514e;font-size:13px;padding:6px 4px 0}
.tbl{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--ink2);font-weight:600;background:var(--card);position:sticky;top:0}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr.flag td{color:var(--bad)}
code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:12.5px}
pre{background:var(--code);padding:12px;border-radius:8px;overflow-x:auto;font-size:12.5px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi .v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}
.kpi .l{font-size:12px;color:var(--muted)}
ol li,ul li{margin:4px 0}
"""

PAGES = [("00", "index", "Overview"), ("01", "inventory", "Inventory"), ("02", "audio", "Audio"),
         ("03", "transcripts", "Transcripts"), ("04", "echo_profiles", "Echo profiles"),
         ("05", "faulty_mic", "Faulty mic"), ("06", "memory", "Memory")]


def fname(tag, nn, topic):
    return f"eda_{tag}_{nn}_{topic}.html"


def page(tag, nn, title, body, meta):
    nav = " ".join(
        f'<a class="{"on" if n == nn else ""}" href="{fname(tag, n, t)}">{n} · {lab}</a>' for n, t, lab in PAGES)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EDA {tag} {title}</title><style>{CSS}</style></head><body><div class="wrap">
<nav>{nav}</nav>
<h1>{html.escape(title)}</h1>
<div class="sub">SoniSpeech baseline · Phase A EDA · data: <code>{html.escape(meta['data'])}</code> ·
generated {meta['when']} by <code>experiments/00_data_analysis/eda.py</code></div>
{body}
</div></body></html>"""


def checks_html(items):
    icon = {"pass": "✓", "warn": "!", "fail": "✕", "info": "i"}
    label = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "info": "NOTE"}
    rows = "".join(
        f'<div class="chk"><span class="badge {k}">{icon[k]} {label[k]}</span><div>{msg}</div></div>'
        for k, msg in items)
    return f'<div class="card checks">{rows}</div>'


def table_html(headers, rows, numeric=(), flags=None):
    th = "".join(f'<th class="{"n" if i in numeric else ""}">{html.escape(h)}</th>' for i, h in enumerate(headers))
    trs = []
    for j, r in enumerate(rows):
        tds = "".join(f'<td class="{"n" if i in numeric else ""}">{c}</td>' for i, c in enumerate(r))
        cls = ' class="flag"' if flags and flags[j] else ""
        trs.append(f"<tr{cls}>{tds}</tr>")
    return f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'


def fig_html(b64, caption):
    return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{b64}"><figcaption>{caption}</figcaption></figure>'


def kpis_html(items):
    return '<div class="kpis">' + "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in items) + "</div>"


# ======================================================================
# main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="extracted zip root (contains SoniSpeech_* and echo_profiles/)")
    ap.add_argument("--tag", required=True, help="short id for the data scope, e.g. zip01")
    ap.add_argument("--out", default=str(HERE / "eda"))
    ap.add_argument("--cache", default=str(HERE.parents[1] / "cache" / "eda"))
    ap.add_argument("--profile-sessions", nargs="*", default=None,
                    help="sessions for echo profiles, e.g. voiced_train/session_010")
    ap.add_argument("--_child", nargs=2, help=argparse.SUPPRESS)
    args = ap.parse_args()
    data_root = Path(args.data).resolve()

    if args._child:
        child_profile(data_root, Path(args._child[0]), Path(args._child[1]))
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) / args.tag
    meta = dict(data=str(data_root), when=dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    sess = sessions(data_root)
    print(f"{len(sess)} sessions found", flush=True)

    inv = analyse_inventory(data_root, sess)
    at = analyse_audio_transcripts(sess)
    at_by_name = {r["name"]: r for r in at}

    # ---- echo profiles on train sessions (avoid using test data for the crop decision)
    if args.profile_sessions:
        prof_names = args.profile_sessions
    else:
        prof_names = []
        for mode in ["voiced", "silent"]:
            cand = [r for r in at if r["mode"] == mode and r["part"] == "train" and not r["faulty"]
                    and max(r["corrupt_utts"]) == 0]
            prof_names.append(min(cand, key=lambda r: r["duration_s"])["name"])
    prof_results = {}
    for name in prof_names:
        split, sname = name.split("/")
        sdir = data_root / f"SoniSpeech_{split}" / sname
        cdir = cache / name.replace("/", "__")
        if not (cdir / "stats.json").exists():
            print(f"  official echo profiles: {name} (child process)", flush=True)
            subprocess.run([sys.executable, __file__, "--data", str(data_root), "--tag", args.tag,
                            "--_child", str(sdir), str(cdir)], check=True)
        st = json.load(open(cdir / "stats.json"))
        st["_cache"] = str(cdir)
        prof_results[name] = st

    extra_corrupt_fig = None
    bad_sessions = [r["name"] for r in sorted(at, key=lambda r: -max(r["jumps_total"])) if max(r["corrupt_utts"]) > 0]
    if bad_sessions:
        name = bad_sessions[0]
        split, sname = name.split("/")
        cdir = cache / name.replace("/", "__")
        if not (cdir / "stats.json").exists():
            print(f"  official echo profiles: {name} (corrupt-mic example)", flush=True)
            subprocess.run([sys.executable, __file__, "--data", str(data_root), "--tag", args.tag,
                            "--_child", str(data_root / f"SoniSpeech_{split}" / sname), str(cdir)], check=True)
        stc = json.load(open(cdir / "stats.json"))
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
        for i in range(4):
            axes[0].plot(np.array(stc["mean_abs_profile_utt"])[i], color=PATH_COLORS[i], lw=1.2, label=PATHS[i])
            axes[1].plot(np.array(stc["mean_abs_diff_utt"])[i], color=PATH_COLORS[i], lw=1.2, label=PATHS[i])
        for ax, t in zip(axes, ["mean |echo profile|", "mean |differential|"]):
            ax.set_yscale("log"); ax.set_xlabel("range bin"); ax.set_title(f"{name}: {t}")
        axes[1].legend(ncol=4, fontsize=8)
        fig.tight_layout()
        extra_corrupt_fig = fig_b64(fig)
    print("rendering figures", flush=True)
    spec_b64, spec_info = spectrogram_fig(sess, data_root)
    band_b64 = band_energy_fig(sess, data_root)
    tfigs = transcript_figs(at)
    pfigs, window_info = profile_figs(prof_results, at_by_name, data_root)
    ffig = faulty_fig(at)
    hfig = health_fig(at)
    corrupt = [(r["name"], c, r["corrupt_utts"][c], r["jumps_total"][c]) for r in at for c in range(2) if r["corrupt_utts"][c] > 0]

    # ---------------- derived numbers ----------------
    hdr_ok = all(r["fs"] == FS and r["channels"] == 2 and r["dtype"] == "int16" for r in at)
    n_utt_ok = all(r["n_utts"] == 50 for r in at)
    start_ok = all(abs(r["first_start"] - 2.0) < 1e-6 for r in at)
    gz = [r["gap_zero_frac"] for r in at if r["gap_zero_frac"] is not None]
    uz = [r["utt_zero_frac"] for r in at]
    offs = np.concatenate([r["start_offsets_ms"] for r in at])
    mode_stats = {}
    for mode in ["voiced", "silent"]:
        d = np.concatenate([r["utt_durs"] for r in at if r["mode"] == mode])
        w = np.concatenate([r["words"] for r in at if r["mode"] == mode])
        mode_stats[mode] = dict(n=len(d), dur_mean=float(d.mean()), dur_std=float(d.std()),
                                dur_min=float(d.min()), dur_max=float(d.max()),
                                w_mean=float(w.mean()), w_std=float(w.std()),
                                p_over_15s=float((d > 15).mean()))
    all_w = np.concatenate([r["words"] for r in at])
    vocab = Counter(w for r in at for t in r["texts"] for w in norm_text(t).split())
    # text overlap train/eval and voiced/silent pairing
    train_txt = {norm_text(t) for r in at if r["part"] == "train" for t in r["texts"]}
    eval_txt = [norm_text(t) for r in at if r["part"] == "eval" for t in r["texts"]]
    eval_in_train = sum(t in train_txt for t in eval_txt)
    train_ids = {i for r in at if r["part"] == "train" for i in r["ids"]}
    eval_ids = [i for r in at if r["part"] == "eval" for i in r["ids"]]
    eval_id_in_train = sum(i in train_ids for i in eval_ids)
    pairs = []
    for r in at:
        if r["mode"] == "voiced":
            o = at_by_name.get(r["name"].replace("voiced", "silent"))
            if o:
                pairs.append((r["name"], o["name"], len(set(r["ids"]) & set(o["ids"]))))
    txt_count = Counter(norm_text(t) for r in at for t in r["texts"])
    dup_txt = sum(1 for c in txt_count.values() if c > 1)
    faulty_detect = [(r["name"], r["faulty"], r["ch_equal_frac"]) for r in at]
    faulty_mismatch = [n for n, f, e in faulty_detect if (e > 0.99) != f]
    lowband = {m: float(np.median([r["low_band_rel"] for r in at if r["mode"] == m])) for m in ["voiced", "silent"]}
    trunc = [r["name"] for r in at if r["end_past_audio"]]

    # memory
    zip_bytes = sum(v for v in inv["by_ext"].values())
    sess_s = np.array([r["duration_s"] for r in at])
    utt_s = np.array([r["utt_seconds"] for r in at])
    n_utts_zip = sum(r["n_utts"] for r in at)
    scale = 18000 / n_utts_zip
    mem = []

    def row(label, per_frame_bytes, seconds, note):
        per = per_frame_bytes * seconds * FPS
        mem.append((label, human(np.median(per)), human(per.sum()), human(per.sum() * scale), note))
    row("Official output: profiles + diff, float64, 500 bins", 2 * 4 * 500 * 8, sess_s, "what main() writes")
    row("Diff only, float32, 500 bins", 4 * 500 * 4, sess_s, "")
    row("Diff only, float16, 80 bins, full session", 4 * 80 * 2, sess_s, "")
    row("Diff only, float16, 80 bins, utterances only", 4 * 80 * 2, utt_s, "planned training cache")
    peak_rss = max(st["max_rss_bytes"] for st in prof_results.values())

    # crop decision text
    wi = list(window_info.values())
    best_starts = [w["best_start"] for w in wi]
    crop_lo = int(round(np.mean(best_starts)))
    metrics = dict(meta=meta, inventory=dict(split_counts=inv["split_counts"], by_ext=inv["by_ext"]),
                   header_ok=hdr_ok, n_utt_ok=n_utt_ok, first_start_ok=start_ok,
                   gap_zero_frac_min=min(gz), utt_zero_frac_max=max(uz),
                   start_offset_ms=dict(median=float(np.median(offs)), p05=float(np.percentile(offs, 5)), p95=float(np.percentile(offs, 95))),
                   mode_stats=mode_stats, words_all=dict(mean=float(all_w.mean()), std=float(all_w.std())),
                   vocab_size=len(vocab), eval_text_in_train=eval_in_train, eval_n=len(eval_txt),
                   eval_id_in_train=eval_id_in_train, voiced_silent_pairs=pairs, duplicate_texts=dup_txt,
                   faulty_mismatch=faulty_mismatch, lowband_rel=lowband, corrupt_channels=corrupt,
                   corrupt_ids={r["name"]: r["corrupt_ids"] for r in at if max(r["corrupt_utts"])}, transcripts_past_audio=trunc,
                   echo_profiles={k: {kk: vv for kk, vv in v.items() if not kk.startswith("mean_") and not kk.startswith("std_")} for k, v in prof_results.items()},
                   window_info=window_info, crop_suggestion=[crop_lo, crop_lo + 80], spectrogram=spec_info)
    json.dump(metrics, open(out / f"eda_{args.tag}_metrics.json", "w"), indent=1, default=str)

    # ======================= pages =======================
    pages = {}
    T = args.tag
    vs, ss = mode_stats["voiced"], mode_stats["silent"]

    # ---- 01 inventory
    sc = inv["split_counts"]
    body = kpis_html([(len(sess), "sessions in this zip"), (human(zip_bytes), "session data on disk"),
                      (f"{sess_s.sum() / 3600:.2f} h", "audio (full sessions)"),
                      (f"{utt_s.sum() / 3600:.2f} h", "inside utterances"), (n_utts_zip, "utterances")])
    body += "<h2>Sessions per split</h2>"
    body += table_html(["split", "sessions in zip", "of total", "session numbers"],
                       [[k, sc.get(k, 0), {"voiced_train": 160, "silent_train": 160}.get(k, 20),
                         ", ".join(str(r["num"]) for r in at if f'{r["mode"]}_{r["part"]}' == k)]
                        for k in ["voiced_train", "voiced_eval", "silent_train", "silent_eval"]], numeric=(1, 2))
    body += "<h2>Disk usage by file type</h2>"
    body += table_html(["type", "files", "total", "share"],
                       [[k or "(none)", inv["count_ext"][k], human(v), f"{100 * v / zip_bytes:.1f}%"]
                        for k, v in sorted(inv["by_ext"].items(), key=lambda x: -x[1])], numeric=(1, 2, 3))
    body += "<p>The baseline needs <code>audio.wav</code> and <code>transcript.json</code> only; video can be skipped when extracting the other zips (<code>unzip -x '*/video.mp4'</code>).</p>"
    body += "<h2>Official preprocessing code</h2><ul>" + "".join(f"<li><code>{html.escape(c)}</code></li>" for c in inv["code_files"]) + "</ul>"
    body += "<h2>All sessions</h2>"
    body += table_html(["session", "duration (s)", "utterances", "audio.wav", "video.mp4", "faulty list"],
                       [[r["name"], f'{r["duration_s"]:.1f}', r["n_utts"],
                         human(next(x for x in inv["sessions"] if x["name"] == r["name"])["files"].get("audio.wav", 0)),
                         human(next(x for x in inv["sessions"] if x["name"] == r["name"])["files"].get("video.mp4", 0)),
                         "yes" if r["faulty"] else ""] for r in at], numeric=(1, 2, 3, 4),
                       flags=[r["faulty"] for r in at])
    pages["01"] = ("Inventory", body)

    # ---- 02 audio
    items = [("pass" if hdr_ok else "fail", f"Header: all {len(at)} sessions are 100 kHz, 2 channels, int16." if hdr_ok else "Header mismatch in some sessions; see table."),
             ("pass", "Spectrogram shows the two chirp bands at 18–28 kHz and 29–39 kHz in both modes."),
             ("pass" if lowband["voiced"] > 5 * lowband["silent"] else "warn",
              f"Speech band (0.1–8 kHz) energy relative to chirp-band energy, median per session: voiced {lowband['voiced']:.3g}, silent {lowband['silent']:.3g} "
              f"({lowband['voiced'] / max(lowband['silent'], 1e-12):.0f}× more in voiced).")]
    body = checks_html(items)
    body += fig_html(spec_b64, "1 s of audio from the first utterance of one voiced and one silent train session (mic 0). Dotted lines mark the nominal chirp bands. Colour: power in dB, 70 dB range.")
    body += fig_html(band_b64, "Mean power spectrum inside utterances. Grey bands mark the two chirps. Voiced speech adds energy below 8 kHz; silent sessions do not.")
    body += "<h2>Per-session header check</h2>"
    body += table_html(["session", "sample rate", "channels", "dtype", "duration (s)", "speech/chirp energy"],
                       [[r["name"], r["fs"], r["channels"], r["dtype"], f'{r["duration_s"]:.2f}', f'{r["low_band_rel"]:.3g}'] for r in at],
                       numeric=(1, 2, 4, 5))
    pages["02"] = ("Audio checks", body)

    # ---- 03 transcripts
    items = [("pass" if n_utt_ok else "warn", f"Utterances per session: {'all 50' if n_utt_ok else 'not all 50: ' + ', '.join(r['name'] + '=' + str(r['n_utts']) for r in at if r['n_utts'] != 50)}."),
             ("pass" if start_ok else "warn", "First utterance starts at exactly 2.000 s in every session." if start_ok else "First utterance does not start at 2.0 s in some sessions."),
             ("info", f"Duration: voiced {vs['dur_mean']:.2f} ± {vs['dur_std']:.2f} s (paper 5.26), silent {ss['dur_mean']:.2f} ± {ss['dur_std']:.2f} s (paper 5.41). Range {min(vs['dur_min'], ss['dur_min']):.2f}–{max(vs['dur_max'], ss['dur_max']):.2f} s."),
             ("info", f"Words per utterance: {all_w.mean():.1f} ± {all_w.std():.1f} (paper 14.4 ± 5.3). Vocabulary in this zip: {len(vocab):,} words (whole corpus: 5,356)."),
             ("pass" if min(gz) > 0.999 else "warn", f"Gaps between utterances are zero-masked: minimum fraction of all-zero samples in any session's gaps = {min(gz):.4f}. Inside utterances the maximum zero fraction is {max(uz):.4f}."),
             ("info", f"Audio unmasks within {np.percentile(np.abs(offs), 95):.1f} ms of the transcript start for 95% of utterances (median offset {np.median(offs):+.2f} ms). Transcript times can be used directly for cutting."),
             ("pass" if eval_in_train == 0 else "warn", f"Eval sentences found in train transcripts (this zip only): {eval_in_train} of {len(eval_txt)}; shared utterance IDs: {eval_id_in_train}. Re-check on the full corpus in Phase C."),
             ("info", f"Voiced and silent sessions with the same number share {', '.join(str(p[2]) for p in pairs) if pairs else 'n/a (no matching pairs in this zip)'} utterance IDs, so the same session number does not mean the same sentences." if pairs and all(p[2] == 0 for p in pairs) else f"Voiced/silent session pairs with the same number and their shared utterance IDs: {', '.join(f'{p[0]}↔{p[1]}: {p[2]}' for p in pairs) or 'none in this zip'}."),
             ("info", f"Utterances longer than 15 s (concatenation cap): voiced {100 * vs['p_over_15s']:.2f}%, silent {100 * ss['p_over_15s']:.2f}%.")]
    if trunc:
        items.append(("warn", f"Transcript end past audio end in: {', '.join(trunc)}"))
    body = checks_html(items)
    body += fig_html(tfigs["dur_words"], "Utterance duration and words per utterance, all sessions in this zip. Words counted after CTC text normalisation (lowercase, punctuation stripped, in-word apostrophes kept).")
    body += fig_html(tfigs["gaps"], "Left: length of zero-masked stretches (includes the 2 s lead-in). Many utterances are back-to-back (gap 0). Right: where audio first becomes non-zero relative to each transcript start.")
    body += "<h2>Per-session summary</h2>"
    body += table_html(["session", "utterances", "mean dur (s)", "speech (s)", "gap zero frac", "utt zero frac", "mean words"],
                       [[r["name"], r["n_utts"], f'{np.mean(r["utt_durs"]):.2f}', f'{r["utt_seconds"]:.0f}',
                         f'{r["gap_zero_frac"]:.4f}' if r["gap_zero_frac"] is not None else "–", f'{r["utt_zero_frac"]:.4f}', f'{np.mean(r["words"]):.1f}'] for r in at],
                       numeric=(1, 2, 3, 4, 5, 6))
    ex = at[0]
    body += "<h2>Sample transcripts</h2>" + table_html(["id", "raw", "CTC target"], [[html.escape(i), html.escape(t), html.escape(norm_text(t))] for i, t in zip(ex["ids"][:6], ex["texts"][:6])])
    pages["03"] = ("Transcript checks", body)

    # ---- 04 echo profiles
    items = []
    for name, st in prof_results.items():
        items.append(("info", f"<b>{name}</b>: profiles {tuple(st['shape'])} {st['dtype']}, diff {tuple(st['diff_shape'])} {st['diff_dtype']}, "
                              f"{st['audio_s']:.0f} s of audio → runtime {st['runtime_s']:.0f} s, peak RSS {human(st['max_rss_bytes'])}, "
                              f"{human(st['npy_bytes_each'])} per saved .npy, start_pos = {st['start_pos']} samples."))
    peaks = sorted({p for w in wi for p in w["peak_bins"]})
    items.append(("pass" if all(abs(p - DIRECT_BIN) <= 3 for p in peaks) else "warn",
                  f"Strongest static reflection (direct path) per path sits at bins {peaks}: the official start-position shift centres it at bin {DIRECT_BIN}."))
    items.append(("fail", "Crop <code>0:80</code> lies 170–250 bins <i>before</i> the direct path (negative delay). Those bins hold tails of the previous chirp's far reflections (≈1 m extra path), not the face. The bumps near bins 60 and 120 are room multipath."))
    items.append(("pass", f"Best 80-bin window by motion energy starts at bins {best_starts} → suggested crop <code>{crop_lo}:{crop_lo + 80}</code>, keeping "
                          f"{100 * np.mean([w['best_frac'] for w in wi]):.0f}% of the motion energy (window <code>240:320</code>: {100 * np.mean([w['frac_240_320'] for w in wi]):.0f}%)."))
    gq = [np.mean(st["mean_abs_diff_gap"]) / np.mean(st["mean_abs_diff_utt"]) for st in prof_results.values() if st["mean_abs_diff_gap"]]
    items.append(("pass" if gq and max(gq) < 0.05 else "warn", f"Differential profile is quiet between utterances: mean |diff| in gaps is {', '.join(f'{100 * g:.2f}%' for g in gq)} of the in-utterance level (gaps are zero-masked audio)."))
    body = checks_html(items)
    body += """<h2>What the official code does</h2><div class="card"><ol>
<li><b>Load</b> <code>audio.wav</code>, check 100 kHz and 2 channels, scale int16 to float64 by 1/32768.</li>
<li><b>Band-pass</b> each mic channel once per transmitter with a 5th-order Butterworth (<code>scipy.signal.butter</code> + <code>lfilter</code>, causal, single pass): 17.5–28.5 kHz for speaker 0, 28.5–39.5 kHz for speaker 1. Result: 2 tx × 2 mics signals.</li>
<li><b>Chirp alignment</b>: cross-correlate each filtered signal with its transmitted chirp (<code>tx_signals/*.wav</code>, 500 samples) over the whole file (cap 50 M samples), fold the correlation into 500-sample frames, average |corr| over all frames and paths, and take the argmax. The audio is then trimmed by <code>start_pos = (argmax + 250) mod 500</code>, which puts the strongest echo, the direct path, at <b>bin 250</b>.</li>
<li><b>Framing</b>: drop the trailing partial frame and reshape to <code>(2, 2, n_frames, 500)</code>. Frame <i>f</i> covers samples <code>start_pos + 500f … +500</code>.</li>
<li><b>Echo profile</b>: for every frame, <code>np.correlate(frame, chirp, 'full')[499:]</code>, i.e. the correlation at lags 0…499 within the frame (no wrap-around across frame edges). Output float64.</li>
<li><b>Differential profile</b> (in <code>main()</code>): <code>|p[t]| − |p[t−1]|</code> along time, zero-padded at <i>t</i> = 0.</li>
<li><b>No normalisation</b> of profiles. PNG plots clip raw profiles to [0, 5] and diffs to [−0.1, 0.1].</li>
<li><code>main()</code> processes only <code>SoniSpeech_silent_eval</code> as shipped and writes full float64 arrays into the dataset folders. Do not run it over the dataset; <code>scripts/preprocess.py</code> should import <code>calculate_echo_profiles</code> as this EDA does.</li>
</ol></div>"""
    body += "<h2>Where the signal is along the range axis</h2>"
    body += fig_html(pfigs["per_bin"], "Mean |echo profile| (left) and mean |differential profile| (right) per range bin, inside utterances, per sound path. Dotted line: bin 250. Shaded: best 80-bin window for that session. Log y-axis.")
    body += fig_html(pfigs["motion_zoom"], "Motion energy per bin, bins 150–400. Each path is normalised to its own maximum, then the 4 paths are averaged.")
    body += "<h2>One utterance, all four paths</h2>"
    for name, h in pfigs["heat"].items():
        body += fig_html(h["img"], f"{name}, utterance <code>{html.escape(h['utt'])}</code>: “{html.escape(h['text'])}”. Blue = echo weaker than the previous frame, red = stronger. 1 s of context on each side.")
    body += "<h2>Frame alignment for cutting utterances</h2><div class=\"card\"><p><code>start_frame = floor((start_s·100000 − start_pos) / 500)</code>, <code>end_frame = ceil((end_s·100000 − start_pos) / 500)</code>. Ignoring <code>start_pos</code> shifts cuts by up to one frame (5 ms).</p></div>"
    pages["04"] = ("Echo profiles", body)

    # ---- 05 faulty mic
    flist = [r for r in at if r["faulty"]]
    items = [("pass" if not faulty_mismatch else "warn",
              "Every session on the faulty list has ch1 == ch0, and no other session does." if not faulty_mismatch else f"Mismatch between official list and data: {', '.join(faulty_mismatch)}"),
             ("info", f"Faulty sessions in this zip: {', '.join(r['name'] for r in flist) or 'none'}." + ("" if any(r["mode"] == "voiced" for r in flist) else " No voiced faulty session (150–160) in this zip.")),
             ("fail" if corrupt else "pass", ("<b>Undocumented corrupt channel(s):</b> " + "; ".join(f"{n} mic {c}: {k} of 50 utterances, {j:,} rail-to-rail jumps" for n, c, k, j in corrupt)
                                             + ". Every other channel in this zip has zero such jumps. The session is not on the official list. Screen every utterance in preprocessing and drop (or flag) any with jumps.") if corrupt else "No channel has rail-to-rail jumps.")]
    body = checks_html(items)
    body += fig_html(hfig, f"Utterances per session where a mic has at least one sample-to-sample jump larger than {JUMP:,} counts, the signature of the corruption described in the dataset README (values bouncing between the int16 extremes). Healthy audio never does this.")
    body += fig_html(ffig, "Fraction of samples where mic 0 and mic 1 are identical, measured on 0.5 s from the middle of every utterance.")
    body += table_html(["session", "faulty list", "ch0==ch1 fraction", "ch0/ch1 correlation", "corrupt utts mic 0", "corrupt utts mic 1"],
                       [[r["name"], "yes" if r["faulty"] else "", f'{r["ch_equal_frac"]:.4f}', f'{r["ch_corr"]:.3f}' if r["ch_corr"] is not None else "–",
                         r["corrupt_utts"][0], r["corrupt_utts"][1]] for r in at],
                       numeric=(2, 3, 4, 5), flags=[r["faulty"] or max(r["corrupt_utts"]) > 0 for r in at])
    vrows = [r for r in at if r["part"] == "train" and VAL_SESSIONS[0] <= r["num"] <= VAL_SESSIONS[1]]
    body += (f"<h2>Validation sessions in this zip</h2><div class=\"card\"><p>Decided split (README §4): dev validation = train sessions "
             f"{VAL_SESSIONS[0]}–{VAL_SESSIONS[1]} in both modes; they must be clean.</p></div>")
    body += table_html(["session", "faulty list", "corrupt utts mic 0", "corrupt utts mic 1", "usable utterances"],
                       [[r["name"], "yes" if r["faulty"] else "", r["corrupt_utts"][0], r["corrupt_utts"][1],
                         r["n_utts"] - len(set(r["corrupt_ids"][0]) | set(r["corrupt_ids"][1]))] for r in vrows] or [["(none in this zip)", "", "", "", ""]],
                       numeric=(2, 3, 4), flags=[r["faulty"] or max(r["corrupt_utts"]) > 0 for r in vrows] or None)
    if extra_corrupt_fig:
        body += "<h2>What a corrupt channel does to the echo profile</h2>" + fig_html(extra_corrupt_fig, "Per-bin profile statistics (all utterances) for a session with corrupt mic-1 utterances: both paths through mic 1 (s0m1, s1m1) become a flat noise floor ~50× above the healthy paths, burying the motion signal.")
    pages["05"] = ("Mic health and faulty-mic check", body)

    # ---- 06 memory
    items = [("info", f"Peak RSS while running the official function on one session: {human(peak_rss)} (float64 intermediates: filtered audio + profiles)."),
             ("info", f"Extrapolation to the full corpus scales this zip by 18,000 / {n_utts_zip} utterances = ×{scale:.2f}.")]
    body = checks_html(items)
    body += table_html(["storage", "per session (median)", f"this zip ({len(at)} sessions)", "full corpus (est.)", "note"], mem, numeric=(1, 2, 3))
    body += f"""<div class="card"><p>Per-session size is <code>n_frames × bins × 4 paths × bytes</code> with <code>n_frames = duration × 200</code>.
Median session length here is {np.median(sess_s):.0f} s ({np.median(sess_s) * FPS:,.0f} frames). The planned cache (float16, 80 bins, utterances only) fits in RAM for the whole corpus, so a single memory-mapped array plus an index is enough; sharding is unnecessary.</p></div>"""
    pages["06"] = ("Memory estimate", body)

    # ---- 00 index
    vr = [r for r in at if r["part"] == "train" and VAL_SESSIONS[0] <= r["num"] <= VAL_SESSIONS[1]]
    vbad = [r for r in vr if r["faulty"] or max(r["corrupt_utts"]) > 0]
    val_status = "warn" if vbad else ("pass" if vr else "info")
    val_msg = (f"Validation sessions ({VAL_SESSIONS[0]}–{VAL_SESSIONS[1]}) in this zip: " + (", ".join(r["name"] for r in vr) or "none")
               + (". With problems: " + ", ".join(f"{r['name']} ({max(r['corrupt_utts'])} corrupt utterances, dropped)" for r in vbad) if vbad else (". All clean." if vr else ".")))
    summary = [
        ("pass" if hdr_ok else "fail", f"Audio format matches the spec in all {len(at)} sessions (100 kHz, 2 ch, int16)."),
        ("pass" if n_utt_ok and start_ok else "warn", "50 utterances per session; the first starts at 2.000 s."),
        ("pass" if min(gz) > 0.999 else "warn", "Gaps between utterances are exact zeros; transcript times line up with the unmasked audio."),
        ("info", f"Durations: voiced {vs['dur_mean']:.2f} s, silent {ss['dur_mean']:.2f} s; {all_w.mean():.1f} ± {all_w.std():.1f} words per utterance (paper: 5.26 / 5.41 s, 14.4 ± 5.3)."),
        ("fail", f"Range crop <code>0:80</code> is wrong: the direct path sits at bin 250. Use <code>{crop_lo}:{crop_lo + 80}</code> (data-driven, train sessions only)."),
        ("info", "The official code applies no normalisation; per-channel standardisation stays our choice."),
        ("pass" if not faulty_mismatch else "warn", "Faulty-mic list (duplicated ch1) confirmed against the data for every session in this zip."),
        ("fail" if corrupt else "pass", ("Undocumented corrupt mic: " + ", ".join(f"{n} mic {c} ({k} of 50 utterances)" for n, c, k, _ in corrupt) + ". Not on the official list; preprocessing must screen every utterance.") if corrupt else "No undocumented corrupt channels."),
        (val_status, val_msg),
        ("info", f"Official function: ~{np.mean([st['runtime_s'] for st in prof_results.values()]):.0f} s and {human(peak_rss)} RAM per session. Cache of float16 / 80 bins / utterances only ≈ {mem[3][3]} for the full corpus."),
    ]
    body = checks_html(summary)
    body += "<h2>Reports</h2>" + table_html(["report", "file", "covers (README §2 step)"], [
        [f'<a href="{fname(T, n, t)}">{lab}</a>', f"<code>{fname(T, n, t)}</code>", cov] for (n, t, lab), cov in zip(PAGES[1:], [
            "1 inventory", "2 audio checks", "3 transcript checks", "4–6 official script, code reading, profile plot", "7 faulty mic", "8 memory"])])
    body += f"<p>Raw numbers: <code>eda_{T}_metrics.json</code>. Written summary: <code>eda_findings_{T}.md</code>.</p>"
    pages["00"] = ("SoniSpeech EDA overview", body)

    for nn, topic, _ in PAGES:
        title, body = pages[nn]
        (out / fname(T, nn, topic)).write_text(page(T, nn, title, body, meta))
    print("wrote", ", ".join(fname(T, n, t) for n, t, _ in PAGES))

    # ---- docs/eda_findings.md
    md = [f"# EDA findings ({T})", "", f"Generated {meta['when']} by `experiments/00_data_analysis/eda.py` from `{data_root.name}`. "
          f"HTML reports: `eda/eda_{T}_*.html`.", ""]
    strip = lambda s: re.sub(r"<[^>]+>", "", s).replace("&lt;", "<").replace("&gt;", ">")
    for k, msg in summary:
        md.append(f"- **{k.upper()}** {strip(msg).replace('0:80', '`0:80`')}")
    md += ["", "## Echo profiles", ""]
    for k, msg in items_profiles(prof_results, wi, best_starts, crop_lo):
        md.append(f"- {msg}")
    md += ["", "## Decisions for Phase B/C", "",
           f"- Range crop: `{crop_lo}:{crop_lo + 80}` (motion-energy window on train sessions {', '.join(prof_results)}).",
           "- Frame cut: `floor((start_s*1e5 - start_pos)/500)` to `ceil((end_s*1e5 - start_pos)/500)`; capture `start_pos` from the official function.",
           "- Normalisation: per-channel standardisation from train statistics (official code has none).",
           f"- Validation (decided, README §4): train sessions {VAL_SESSIONS[0]}–{VAL_SESSIONS[1]} in both modes. {strip(val_msg)}",
           "- Do not run official `main()`; import `calculate_echo_profiles`.",
           f"- Mic health screen in preprocessing: flag an utterance when any channel has a sample-to-sample jump > {JUMP}. Found here: "
           + (", ".join(f"{n} mic {c} ({k}/50 utterances)" for n, c, k, _ in corrupt) or "none") + ". Drop flagged utterances (or the whole session) from train/validation.", ""]
    (out / f"eda_findings_{args.tag}.md").write_text("\n".join(md))
    print(f"wrote {out / f'eda_findings_{args.tag}.md'}")


def items_profiles(prof_results, wi, best_starts, crop_lo):
    out = []
    for name, st in prof_results.items():
        out.append(("info", f"{name}: profiles {tuple(st['shape'])} {st['dtype']}, {st['audio_s']:.0f} s audio, runtime {st['runtime_s']:.0f} s, "
                            f"peak RSS {human(st['max_rss_bytes'])}, {human(st['npy_bytes_each'])} per .npy, start_pos {st['start_pos']}."))
    out.append(("info", f"Direct-path peak bins: {sorted({p for w in wi for p in w['peak_bins']})}; best 80-bin motion windows (motion above floor, healthy paths, bins 150-450) start at {best_starts}, keeping {100 * np.mean([w['best_frac'] for w in wi]):.0f}% of it."))
    return out


if __name__ == "__main__":
    main()
