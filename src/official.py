"""Thin wrapper around the dataset's official echo_profiles code (shipped in zip 01).

The official calculate_echo_profiles() is called unchanged. Its internal start_pos
(the chirp alignment offset, 0-499 samples) is needed to convert transcript times to
frame indices, so it is captured from the function's locals with a return trace
instead of re-implementing anything.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

TX_FILES = ["fmcw18000_b10000_l500_100k.wav", "fmcw29000_b10000_l500_100k.wav"]
BP_RANGES = [[17500, 28500], [28500, 39500]]  # as in the official main()

_ep = None
_tx = None


def _load(code_dir: Path):
    global _ep, _tx
    if _ep is None:
        sys.path.insert(0, str(code_dir))
        import echo_profiles as ep  # noqa: E402  (official module)
        _ep = ep
        _tx = []
        for f in TX_FILES:
            _, tx = wavfile.read(code_dir / "tx_signals" / f)
            _tx.append(tx.astype(float) / 32768.0)
    return _ep, _tx


def find_code_dir(*roots: Path) -> Path:
    """Locate echo_profiles/ in any given root or its SoniSpeech_* siblings."""
    for root in roots:
        root = Path(root)
        for cand in [root / "echo_profiles", *sorted(root.glob("SoniSpeech_*/echo_profiles")),
                     *sorted(root.parent.glob("SoniSpeech_*/echo_profiles"))]:
            if (cand / "echo_profiles.py").exists():
                return cand
    raise FileNotFoundError("official echo_profiles/ code not found (it ships in zip 01)")


def echo_profiles(code_dir: Path, audio_path: Path):
    """Return (profiles (2, 2, n_frames, 500) float64, start_pos int)."""
    ep, tx = _load(Path(code_dir))
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

    old = sys.gettrace()
    sys.settrace(gtrace)
    try:
        profiles = ep.calculate_echo_profiles(str(audio_path), tx, BP_RANGES)
    finally:
        sys.settrace(old)
    return profiles, captured["start_pos"]


def diff_crop(profiles: np.ndarray, crop) -> np.ndarray:
    """Official differential profile |p_t| - |p_(t-1)| (zero first frame), reshaped to
    4 paths [s0m0, s0m1, s1m0, s1m1] and cropped to range bins crop[0]:crop[1].

    The difference is element-wise per bin, so cropping first gives identical values to
    the official main() (diff over all 500 bins, then crop) with 6x less memory.
    Returns (4, n_frames, n_bins) float32.
    """
    n_tx, n_mic, n_frames, _ = profiles.shape
    a = np.abs(profiles.reshape(n_tx * n_mic, n_frames, -1)[:, :, crop[0]:crop[1]])
    d = np.zeros_like(a, dtype=np.float32)
    d[:, 1:] = a[:, 1:] - a[:, :-1]
    return d
