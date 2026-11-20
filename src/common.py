"""Constants and rules shared by preprocessing, tokenizer, data loading and evaluation.

Everything here was settled in Phase A (see README sections 1, 3, 4 and eda/).
"""
import re

FS = 100_000          # audio sample rate
FRAME = 500           # samples per chirp = samples per echo frame
FPS = FS // FRAME     # 200 frames per second
CROP = (226, 306)     # range bins kept (80); direct path sits at bin 250
PATHS = ["s0m0", "s0m1", "s1m0", "s1m1"]  # channel order = (speaker, mic) flattened
JUMP = 30_000         # |x[n] - x[n-1]| above this = rail-to-rail corruption / transient

# Official faulty-mic list (mic 1 replaced by a copy of mic 0). Never used for training.
FAULTY = {("voiced", "train"): set(range(150, 161)),
          ("silent", "train"): set(range(141, 161))}

DEV_TRAIN_MAX = 131      # dev split: train 1-131, validate 132-140
DEV_VAL = range(132, 141)
FINAL_TRAIN_MAX = 140    # final split: train 1-140 (the paper's setup)

# Transcript typo found in EDA (silent_train 078, train_03881).
TEXT_FIXES = {"thatDad": "that Dad"}


def fix_text(t: str) -> str:
    for bad, good in TEXT_FIXES.items():
        t = t.replace(bad, good)
    return t


def norm_text(t: str) -> str:
    """CTC target text: lowercase, punctuation stripped, in-word apostrophes kept."""
    t = t.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", t)
    return " ".join(t.split())


def split_labels(mode: str, part: str, num: int):
    """Return (dev_split, final_split), each one of train / val / test / unused."""
    if part == "eval":
        return "test", "test"
    dev = "train" if num <= DEV_TRAIN_MAX else ("val" if num in DEV_VAL else "unused")
    final = "train" if num <= FINAL_TRAIN_MAX else "unused"
    return dev, final


def is_faulty(mode: str, part: str, num: int) -> bool:
    return num in FAULTY.get((mode, part), set())
