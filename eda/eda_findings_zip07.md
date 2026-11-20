# EDA findings (zip07)

Generated 2026-11-20 04:46 by `scripts/eda.py` from `SoniSpeech_07_of_09`. HTML reports: `eda/eda_zip07_*.html`.

- **PASS** Audio format matches the spec in all 40 sessions (100 kHz, 2 ch, int16).
- **PASS** 50 utterances per session; the first starts at 2.000 s.
- **PASS** Gaps between utterances are exact zeros; transcript times line up with the unmasked audio.
- **INFO** Durations: voiced 5.34 s, silent 5.40 s; 14.4 ± 5.3 words per utterance (paper: 5.26 / 5.41 s, 14.4 ± 5.3).
- **FAIL** Range crop `0:80` is wrong: the direct path sits at bin 250. Use 223:303 (data-driven, train sessions only).
- **INFO** The official code applies no normalisation; per-channel standardisation stays our choice.
- **PASS** Faulty-mic list (duplicated ch1) confirmed against the data for every session in this zip.
- **PASS** No undocumented corrupt channels.
- **PASS** Validation sessions (132–140) in this zip: voiced_train/session_132, silent_train/session_140. All clean.
- **INFO** Official function: ~14 s and 3.8 GB RAM per session. Cache of float16 / 80 bins / utterances only ≈ 11.5 GB for the full corpus.

## Echo profiles

- voiced_train/session_079: profiles (2, 2, 58739, 500) float64, 294 s audio, runtime 15 s, peak RSS 3.4 GB, 896.3 MB per .npy, start_pos 107.
- silent_train/session_050: profiles (2, 2, 56531, 500) float64, 283 s audio, runtime 14 s, peak RSS 3.8 GB, 862.6 MB per .npy, start_pos 138.
- Direct-path peak bins: [226, 227, 234, 236, 241, 250, 252]; best 80-bin motion windows (motion above floor, healthy paths, bins 150-450) start at [223, 223], keeping 74% of it.

## Decisions for Phase B/C

- Range crop: `223:303` (motion-energy window on train sessions voiced_train/session_079, silent_train/session_050).
- Frame cut: `floor((start_s*1e5 - start_pos)/500)` to `ceil((end_s*1e5 - start_pos)/500)`; capture `start_pos` from the official function.
- Normalisation: per-channel standardisation from train statistics (official code has none).
- Validation (decided, README §4): train sessions 132–140 in both modes. Validation sessions (132–140) in this zip: voiced_train/session_132, silent_train/session_140. All clean.
- Do not run official `main()`; import `calculate_echo_profiles`.
- Mic health screen in preprocessing: flag an utterance when any channel has a sample-to-sample jump > 30000. Found here: none. Drop flagged utterances (or the whole session) from train/validation.
