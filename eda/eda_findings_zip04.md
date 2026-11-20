# EDA findings (zip04)

Generated 2026-11-19 15:40 by `scripts/eda.py` from `SoniSpeech_04_of_09`. HTML reports: `eda/eda_zip04_*.html`.

- **PASS** Audio format matches the spec in all 40 sessions (100 kHz, 2 ch, int16).
- **PASS** 50 utterances per session; the first starts at 2.000 s.
- **PASS** Gaps between utterances are exact zeros; transcript times line up with the unmasked audio.
- **INFO** Durations: voiced 5.21 s, silent 5.44 s; 14.3 ± 5.4 words per utterance (paper: 5.26 / 5.41 s, 14.4 ± 5.3).
- **FAIL** Range crop `0:80` is wrong: the direct path sits at bin 250. Use 224:304 (data-driven, train sessions only).
- **INFO** The official code applies no normalisation; per-channel standardisation stays our choice.
- **PASS** Faulty-mic list (duplicated ch1) confirmed against the data for every session in this zip.
- **PASS** No undocumented corrupt channels.
- **PASS** Validation sessions (132–140) in this zip: voiced_train/session_137. All clean.
- **INFO** Official function: ~13 s and 3.9 GB RAM per session. Cache of float16 / 80 bins / utterances only ≈ 11.4 GB for the full corpus.

## Echo profiles

- voiced_train/session_119: profiles (2, 2, 56949, 500) float64, 285 s audio, runtime 13 s, peak RSS 3.9 GB, 869.0 MB per .npy, start_pos 77.
- silent_train/session_124: profiles (2, 2, 56565, 500) float64, 283 s audio, runtime 12 s, peak RSS 3.9 GB, 863.1 MB per .npy, start_pos 301.
- Direct-path peak bins: [228, 230, 234, 236, 241, 250, 252]; best 80-bin motion windows (motion above floor, healthy paths, bins 150-450) start at [225, 224], keeping 76% of it.

## Decisions for Phase B/C

- Range crop: `224:304` (motion-energy window on train sessions voiced_train/session_119, silent_train/session_124).
- Frame cut: `floor((start_s*1e5 - start_pos)/500)` to `ceil((end_s*1e5 - start_pos)/500)`; capture `start_pos` from the official function.
- Normalisation: per-channel standardisation from train statistics (official code has none).
- Validation (decided, README §4): train sessions 132–140 in both modes. Validation sessions (132–140) in this zip: voiced_train/session_137. All clean.
- Do not run official `main()`; import `calculate_echo_profiles`.
- Mic health screen in preprocessing: flag an utterance when any channel has a sample-to-sample jump > 30000. Found here: none. Drop flagged utterances (or the whole session) from train/validation.
