# eda/

Phase A exploratory data analysis: HTML reports, raw metrics and written findings, one set per zip. Reproducible with `scripts/eda.py`.

## Naming convention

```
eda/eda_<scope>_<NN>_<topic>.html    reports; NN = 00 overview … 06
eda/eda_<scope>_metrics.json         raw numbers
eda/eda_findings_<scope>.md          written summary
```

`scope` is `zip01` … `zip09`, or `full` for the whole corpus.

## Whole-corpus files (from `scripts/corpus_check.py`, run after all zips)

| File | Content |
| --- | --- |
| `eda_findings_full.md` | Completeness, drops, transcript differences, crop |
| `eda_full_metrics.json` | Raw numbers |
| `drop_list.csv` | Utterances Phase B excludes from train/validation, with reason |

## Current contents

Reports exist for `zip01` to `zip09`; the table lists zip01, and the others have the same files under their own tag.

| File | Content |
| --- | --- |
| `eda_zip01_00_index.html` | Overview: all checks with pass/warn/fail, links to the rest |
| `eda_zip01_01_inventory.html` | Splits, sessions, disk usage by file type |
| `eda_zip01_02_audio.html` | Header checks, spectrograms, speech-band energy |
| `eda_zip01_03_transcripts.html` | Utterance counts, durations, words, zero-masking, text overlap |
| `eda_zip01_04_echo_profiles.html` | Official code walkthrough, runtime/memory, range-bin crop, utterance heatmaps |
| `eda_zip01_05_faulty_mic.html` | Duplicated-channel check and corrupt-channel screen |
| `eda_zip01_06_memory.html` | Storage estimates per session, zip and corpus |
| `eda_zip01_metrics.json` | Raw numbers behind the reports |

Regenerate (from the project root): `.venv/bin/python scripts/eda.py --data data/SoniSpeech_0N_of_09 --tag zip0N`
(zips without `echo_profiles/` use zip01's copy). 
(about 1 min; echo-profile runs are cached in `cache/eda/`).
