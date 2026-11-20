# EDA findings (zip01)

Generated 2026-11-19 15:38 by `scripts/eda.py` from `SoniSpeech_01_of_09`. HTML reports: `eda/eda_zip01_*.html`.

- **PASS** Audio format matches the spec in all 40 sessions (100 kHz, 2 ch, int16).
- **PASS** 50 utterances per session; the first starts at 2.000 s.
- **PASS** Gaps between utterances are exact zeros; transcript times line up with the unmasked audio.
- **INFO** Durations: voiced 5.23 s, silent 5.38 s; 14.3 ± 5.3 words per utterance (paper: 5.26 / 5.41 s, 14.4 ± 5.3).
- **FAIL** Range crop `0:80` is wrong: the direct path sits at bin 250. Use 229:309 (data-driven, train sessions only).
- **INFO** The official code applies no normalisation; per-channel standardisation stays our choice.
- **PASS** Faulty-mic list (duplicated ch1) confirmed against the data for every session in this zip.
- **FAIL** Undocumented corrupt mic: voiced_train/session_010 mic 1 (1 of 50 utterances), voiced_train/session_135 mic 1 (18 of 50 utterances). Not on the official list; preprocessing must screen every utterance.
- **WARN** Validation sessions (132–140) in this zip: voiced_train/session_135, silent_train/session_132, silent_train/session_134, silent_train/session_139. With problems: voiced_train/session_135 (18 corrupt utterances, dropped)
- **INFO** Official function: ~13 s and 4.3 GB RAM per session. Cache of float16 / 80 bins / utterances only ≈ 11.4 GB for the full corpus.

## Echo profiles

- voiced_train/session_123: profiles (2, 2, 59235, 500) float64, 296 s audio, runtime 14 s, peak RSS 3.0 GB, 903.9 MB per .npy, start_pos 447.
- silent_train/session_094: profiles (2, 2, 55777, 500) float64, 279 s audio, runtime 12 s, peak RSS 4.3 GB, 851.1 MB per .npy, start_pos 46.
- Direct-path peak bins: [235, 241, 248, 257]; best 80-bin motion windows (motion above floor, healthy paths, bins 150-450) start at [228, 230], keeping 74% of it.

## Decisions for Phase B/C

- Range crop: `229:309` (motion-energy window on train sessions voiced_train/session_123, silent_train/session_094).
- Frame cut: `floor((start_s*1e5 - start_pos)/500)` to `ceil((end_s*1e5 - start_pos)/500)`; capture `start_pos` from the official function.
- Normalisation: per-channel standardisation from train statistics (official code has none).
- Validation (decided, README §4): train sessions 132–140 in both modes. Validation sessions (132–140) in this zip: voiced_train/session_135, silent_train/session_132, silent_train/session_134, silent_train/session_139. With problems: voiced_train/session_135 (18 corrupt utterances, dropped)
- Do not run official `main()`; import `calculate_echo_profiles`.
- Mic health screen in preprocessing: flag an utterance when any channel has a sample-to-sample jump > 30000. Found here: voiced_train/session_010 mic 1 (1/50 utterances), voiced_train/session_135 mic 1 (18/50 utterances). Drop flagged utterances (or the whole session) from train/validation.
