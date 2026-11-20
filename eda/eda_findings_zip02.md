# EDA findings (zip02)

Generated 2026-11-19 15:38 by `scripts/eda.py` from `SoniSpeech_02_of_09`. HTML reports: `eda/eda_zip02_*.html`.

- **PASS** Audio format matches the spec in all 40 sessions (100 kHz, 2 ch, int16).
- **PASS** 50 utterances per session; the first starts at 2.000 s.
- **PASS** Gaps between utterances are exact zeros; transcript times line up with the unmasked audio.
- **INFO** Durations: voiced 5.27 s, silent 5.44 s; 14.5 ± 5.4 words per utterance (paper: 5.26 / 5.41 s, 14.4 ± 5.3).
- **FAIL** Range crop `0:80` is wrong: the direct path sits at bin 250. Use 226:306 (data-driven, train sessions only).
- **INFO** The official code applies no normalisation; per-channel standardisation stays our choice.
- **PASS** Faulty-mic list (duplicated ch1) confirmed against the data for every session in this zip.
- **PASS** No undocumented corrupt channels.
- **PASS** Validation sessions (132–140) in this zip: voiced_train/session_133, voiced_train/session_134, voiced_train/session_140. All clean.
- **INFO** Official function: ~13 s and 4.4 GB RAM per session. Cache of float16 / 80 bins / utterances only ≈ 11.5 GB for the full corpus.

## Echo profiles

- voiced_train/session_076: profiles (2, 2, 58409, 500) float64, 292 s audio, runtime 13 s, peak RSS 3.3 GB, 891.3 MB per .npy, start_pos 88.
- silent_train/session_086: profiles (2, 2, 60142, 500) float64, 301 s audio, runtime 13 s, peak RSS 4.4 GB, 917.7 MB per .npy, start_pos 420.
- Direct-path peak bins: [227, 230, 236, 238, 239, 243, 252, 257]; best 80-bin motion windows (motion above floor, healthy paths, bins 150-450) start at [223, 230], keeping 76% of it.

## Decisions for Phase B/C

- Range crop: `226:306` (motion-energy window on train sessions voiced_train/session_076, silent_train/session_086).
- Frame cut: `floor((start_s*1e5 - start_pos)/500)` to `ceil((end_s*1e5 - start_pos)/500)`; capture `start_pos` from the official function.
- Normalisation: per-channel standardisation from train statistics (official code has none).
- Validation (decided, README §4): train sessions 132–140 in both modes. Validation sessions (132–140) in this zip: voiced_train/session_133, voiced_train/session_134, voiced_train/session_140. All clean.
- Do not run official `main()`; import `calculate_echo_profiles`.
- Mic health screen in preprocessing: flag an utterance when any channel has a sample-to-sample jump > 30000. Found here: none. Drop flagged utterances (or the whole session) from train/validation.
