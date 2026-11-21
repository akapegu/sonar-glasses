# 00 Data analysis

Exploratory analysis of the nine SoniSpeech zips, the whole-corpus checks that produced the
drop list, and an HTML explainer of the data and preprocessing.

| File | What it does |
| --- | --- |
| `eda.py` | Per-zip EDA: HTML reports, metrics JSON and a findings summary in `eda/` |
| `corpus_check.py` | Whole-corpus checks after all zips: completeness, corruption, overlaps; writes `eda/drop_list.csv` |
| `make_explainer_figs.py` | Renders the real-data figures for `explainer/data_and_preprocessing.html` |
| `eda/` | Reports (`eda_zipNN_*.html`), findings (`eda_findings_*.md`), `drop_list.csv` (read by `scripts/preprocess.py`) |
| `explainer/` | Data and preprocessing explainer with flowcharts |

```bash
.venv/bin/python experiments/00_data_analysis/eda.py --data data/SoniSpeech_01_of_09 --tag zip01
.venv/bin/python experiments/00_data_analysis/corpus_check.py
.venv/bin/python experiments/00_data_analysis/make_explainer_figs.py
```

## Main findings

- 360 sessions, 18,000 utterances, 26.7 h. Voiced session N and silent session N share their 50 sentences.
- The official faulty-mic list is right: voiced 150–160 and silent 141–160 have mic 1 copied from mic 0.
- Undocumented corruption (rail-to-rail jumps > 30000 on mic 1): voiced 135 (18 utterances),
  silent 126 (8), voiced 149 (3), voiced 004 and 010 (1 each). Plus a transient in silent 001
  and one training sentence that also appears in test. 34 utterances dropped in total.
- The official code puts the direct path at range bin 250. The crop is frozen at bins 226:306
  (best motion window across 18 clean sessions).
- One transcript typo ("thatDad", silent 078) is fixed in `src/common.py`.

Details: `eda/eda_findings_full.md` and [docs/reproduction_spec.md](../../docs/reproduction_spec.md).
