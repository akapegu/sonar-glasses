# EDA findings (zip05)

Generated 2026-11-20 02:36 by `scripts/eda.py` from `SoniSpeech_05_of_09`. HTML reports: `eda/eda_zip05_*.html`.

- **PASS** Audio format matches the spec in all 40 sessions (100 kHz, 2 ch, int16).
- **PASS** 50 utterances per session; the first starts at 2.000 s.
- **PASS** Gaps between utterances are exact zeros; transcript times line up with the unmasked audio.
- **INFO** Durations: voiced 5.19 s, silent 5.47 s; 14.3 ± 5.3 words per utterance (paper: 5.26 / 5.41 s, 14.4 ± 5.3).
- **FAIL** Range crop `0:80` is wrong: the direct path sits at bin 250. Use 225:305 (data-driven, train sessions only).
- **INFO** The official code applies no normalisation; per-channel standardisation stays our choice.
- **PASS** Faulty-mic list (duplicated ch1) confirmed against the data for every session in this zip.
- **FAIL** Undocumented corrupt mic: voiced_train/session_004 mic 1 (1 of 50 utterances), silent_train/session_126 mic 1 (8 of 50 utterances). Not on the official list; preprocessing must screen every utterance.
- **INFO** Validation sessions (132–140) in this zip: none.
- **INFO** Official function: ~12 s and 3.9 GB RAM per session. Cache of float16 / 80 bins / utterances only ≈ 11.4 GB for the full corpus.

## Echo profiles

- voiced_train/session_013: profiles (2, 2, 54254, 500) float64, 271 s audio, runtime 12 s, peak RSS 3.9 GB, 827.9 MB per .npy, start_pos 170.
- silent_train/session_043: profiles (2, 2, 58851, 500) float64, 294 s audio, runtime 12 s, peak RSS 3.7 GB, 898.0 MB per .npy, start_pos 351.
- Direct-path peak bins: [227, 232, 238, 239, 252]; best 80-bin motion windows (motion above floor, healthy paths, bins 150-450) start at [225, 225], keeping 75% of it.

## Decisions for Phase B/C

- Range crop: `225:305` (motion-energy window on train sessions voiced_train/session_013, silent_train/session_043).
- Frame cut: `floor((start_s*1e5 - start_pos)/500)` to `ceil((end_s*1e5 - start_pos)/500)`; capture `start_pos` from the official function.
- Normalisation: per-channel standardisation from train statistics (official code has none).
- Validation (decided, README §4): train sessions 132–140 in both modes. Validation sessions (132–140) in this zip: none.
- Do not run official `main()`; import `calculate_echo_profiles`.
- Mic health screen in preprocessing: flag an utterance when any channel has a sample-to-sample jump > 30000. Found here: voiced_train/session_004 mic 1 (1/50 utterances), silent_train/session_126 mic 1 (8/50 utterances). Drop flagged utterances (or the whole session) from train/validation.
