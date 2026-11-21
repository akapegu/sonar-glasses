# 04 LM decoding (improvements phase A)

Beam search with a word 4-gram language model over the saved CTC outputs. No retraining.

| File | What it does |
| --- | --- |
| `../../scripts/dump_logprobs.py` | Saves per-utterance log-probs `(T', 1001)` + references (run on a GPU pod; `pod_pipeline.sh` does it after every run) |
| `lm_decode.py lm` | Builds a KenLM 4-gram from training transcripts (optionally + external text) |
| `lm_decode.py tune` | Grid over alpha (LM weight) and beta (word bonus) on dev validation |
| `lm_decode.py decode` | Decodes once with frozen settings; saves top-10 beams for phase B |
| `build_external_text.py` | SODA train + Cornell Movie Dialogs, normalised like the corpus, leakage-filtered |

Needs the decoding environment (`requirements-decode.txt`: pyctcdecode needs numpy < 2) and
KenLM's `lmplz` / `build_binary` (see [docs/runpod.md](../../docs/runpod.md)).

```bash
V=/workspace/venv-decode/bin/python; X=experiments/04_lm_decoding
$V $X/lm_decode.py lm --cache /workspace/cache/v1 --split dev --out /workspace/lm/dev
$V $X/lm_decode.py tune --logprobs /workspace/runs/baseline_vs_dev/logprobs_val_best \
    --tokenizer /workspace/runs/baseline_vs_dev/tokenizer.model --lm /workspace/lm/dev
$V $X/lm_decode.py lm --cache /workspace/cache/v1 --split final --out /workspace/lm/final
$V $X/lm_decode.py decode --logprobs /workspace/runs/final_vs_final/logprobs_test_last \
    --tokenizer /workspace/runs/final_vs_final/tokenizer.model --lm /workspace/lm/final --alpha 0.5 --beta 2.0
```

## Results: in-domain LM

LM: the 7,002 unique training sentences of sessions 1–140 (4,791 word types; SoniGAN's
lexicon has 4,789). Settings tuned on dev validation with `baseline_vs_dev` and an LM from
sessions 1–131: alpha 0.5, beta 2.0, beam 128 (beam 512 adds nothing; the optimum is flat over
alpha 0.35–0.5 and beta 1–2). Dev validation: 25.5 / 31.8 → 20.1 / 25.6.

Test, scored once (Whisper, ours in brackets):

| Model | Decoder | Voiced | Silent | OOV recall (V / S) |
| --- | --- | --- | --- | --- |
| final_vs | greedy | 18.1% (19.2%) | 29.4% (30.6%) | 1.1% / 0.4% |
| final_vs | **beam + 4-gram** | **15.1%** (16.0%) | **24.2%** (25.4%) | 0.4% / 0.0% |
| baseline_vs_dev | greedy | 21.2% (22.3%) | 32.5% (34.0%) | 0.7% / 0.3% |
| baseline_vs_dev | beam + 4-gram | 17.1% (18.1%) | 26.7% (28.0%) | 0.0% / 0.0% |

With the LM, silent test WER (24.2%) is below the paper's 26.3%, but the paper uses no LM.
Perfect sentences rise from 23.0% to 29.1% (voiced) and 9.5% to 15.4% (silent). OOV recall is
about zero even with greedy decoding: the model does not spell words it never saw.

## External text (phase A2.2)

`build_external_text.py` keeps dialogue turns of 4–40 words from SODA's *train* split and
Cornell Movie Dialogs (OpenSubtitles was skipped) and drops any turn that equals a test or
validation sentence or shares an 8-word sequence with one.

| Source | Turns | Removed (exact) | Removed (8-gram) | Kept |
| --- | --- | --- | --- | --- |
| SODA train | 9,081,257 | 7,136 | 206,090 | 8,063,795 |
| Cornell Movie Dialogs | 304,713 | 8 | 33 | 222,892 |

The filter is essential: **203 of the 1,000 test sentences appear verbatim in SODA's train
split** (6,593 times), and 168,406 turns share an 8-gram with a test sentence. An unfiltered
SODA LM would leak a fifth of the test set.

External LM: in-domain transcripts repeated ×100 plus the 8.29 M filtered turns, 3- and 4-gram
singletons pruned, 40,827-word unigram list. Results: to be filled in (tuning on dev validation).
