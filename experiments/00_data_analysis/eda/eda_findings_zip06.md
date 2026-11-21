# EDA findings (zip06)

Generated 2026-11-20 02:38 by `scripts/eda.py` from `SoniSpeech_06_of_09`. HTML reports: `eda/eda_zip06_*.html`.

- **PASS** Audio format matches the spec in all 40 sessions (100 kHz, 2 ch, int16).
- **PASS** 50 utterances per session; the first starts at 2.000 s.
- **PASS** Gaps between utterances are exact zeros; transcript times line up with the unmasked audio.
- **INFO** Durations: voiced 5.31 s, silent 5.38 s; 14.4 ± 5.4 words per utterance (paper: 5.26 / 5.41 s, 14.4 ± 5.3).
- **FAIL** Range crop `0:80` is wrong: the direct path sits at bin 250. Use 226:306 (data-driven, train sessions only).
- **INFO** The official code applies no normalisation; per-channel standardisation stays our choice.
- **PASS** Faulty-mic list (duplicated ch1) confirmed against the data for every session in this zip.
- **PASS** No undocumented corrupt channels.
- **PASS** Validation sessions (132–140) in this zip: voiced_train/session_136, voiced_train/session_139, silent_train/session_133. All clean.
- **INFO** Official function: ~12 s and 5.4 GB RAM per session. Cache of float16 / 80 bins / utterances only ≈ 11.5 GB for the full corpus.

## Echo profiles

- voiced_train/session_136: profiles (2, 2, 60230, 500) float64, 301 s audio, runtime 13 s, peak RSS 4.8 GB, 919.0 MB per .npy, start_pos 389.
- silent_train/session_121: profiles (2, 2, 54154, 500) float64, 271 s audio, runtime 12 s, peak RSS 5.4 GB, 826.3 MB per .npy, start_pos 84.
- Direct-path peak bins: [228, 230, 234, 236, 239, 241, 250, 252]; best 80-bin motion windows (motion above floor, healthy paths, bins 150-450) start at [228, 223], keeping 72% of it.

## Decisions for Phase B/C

- Range crop: `226:306` (motion-energy window on train sessions voiced_train/session_136, silent_train/session_121).
- Frame cut: `floor((start_s*1e5 - start_pos)/500)` to `ceil((end_s*1e5 - start_pos)/500)`; capture `start_pos` from the official function.
- Normalisation: per-channel standardisation from train statistics (official code has none).
- Validation (decided, README §4): train sessions 132–140 in both modes. Validation sessions (132–140) in this zip: voiced_train/session_136, voiced_train/session_139, silent_train/session_133. All clean.
- Do not run official `main()`; import `calculate_echo_profiles`.
- Mic health screen in preprocessing: flag an utterance when any channel has a sample-to-sample jump > 30000. Found here: none. Drop flagged utterances (or the whole session) from train/validation.
