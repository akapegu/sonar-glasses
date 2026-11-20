# EDA findings (zip08)

Generated 2026-11-20 05:08 by `scripts/eda.py` from `SoniSpeech_08_of_09`. HTML reports: `eda/eda_zip08_*.html`.

- **PASS** Audio format matches the spec in all 40 sessions (100 kHz, 2 ch, int16).
- **PASS** 50 utterances per session; the first starts at 2.000 s.
- **PASS** Gaps between utterances are exact zeros; transcript times line up with the unmasked audio.
- **INFO** Durations: voiced 5.27 s, silent 5.40 s; 14.4 ± 5.3 words per utterance (paper: 5.26 / 5.41 s, 14.4 ± 5.3).
- **FAIL** Range crop `0:80` is wrong: the direct path sits at bin 250. Use 228:308 (data-driven, train sessions only).
- **INFO** The official code applies no normalisation; per-channel standardisation stays our choice.
- **PASS** Faulty-mic list (duplicated ch1) confirmed against the data for every session in this zip.
- **FAIL** Undocumented corrupt mic: voiced_train/session_149 mic 1 (3 of 50 utterances), silent_train/session_001 mic 0 (1 of 50 utterances), silent_train/session_001 mic 1 (1 of 50 utterances). Not on the official list; preprocessing must screen every utterance.
- **PASS** Validation sessions (132–140) in this zip: silent_train/session_137, silent_train/session_138. All clean.
- **INFO** Official function: ~13 s and 5.1 GB RAM per session. Cache of float16 / 80 bins / utterances only ≈ 11.5 GB for the full corpus.

## Echo profiles

- voiced_train/session_017: profiles (2, 2, 65767, 500) float64, 329 s audio, runtime 15 s, peak RSS 4.2 GB, 1,003.5 MB per .npy, start_pos 155.
- silent_train/session_022: profiles (2, 2, 56398, 500) float64, 282 s audio, runtime 12 s, peak RSS 5.1 GB, 860.6 MB per .npy, start_pos 336.
- Direct-path peak bins: [232, 234, 238, 239, 240, 248, 252, 259]; best 80-bin motion windows (motion above floor, healthy paths, bins 150-450) start at [225, 232], keeping 75% of it.

## Decisions for Phase B/C

- Range crop: `228:308` (motion-energy window on train sessions voiced_train/session_017, silent_train/session_022).
- Frame cut: `floor((start_s*1e5 - start_pos)/500)` to `ceil((end_s*1e5 - start_pos)/500)`; capture `start_pos` from the official function.
- Normalisation: per-channel standardisation from train statistics (official code has none).
- Validation (decided, README §4): train sessions 132–140 in both modes. Validation sessions (132–140) in this zip: silent_train/session_137, silent_train/session_138. All clean.
- Do not run official `main()`; import `calculate_echo_profiles`.
- Mic health screen in preprocessing: flag an utterance when any channel has a sample-to-sample jump > 30000. Found here: voiced_train/session_149 mic 1 (3/50 utterances), silent_train/session_001 mic 0 (1/50 utterances), silent_train/session_001 mic 1 (1/50 utterances). Drop flagged utterances (or the whole session) from train/validation.
