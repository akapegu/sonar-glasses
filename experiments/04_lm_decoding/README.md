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
singletons pruned, 40,827-word unigram list (dev version; built in about 15 min on 28 CPUs).

Dev validation (`baseline_vs_dev`, Whisper WER, ours in brackets):

| LM | Best alpha / beta | Voiced | Silent |
| --- | --- | --- | --- |
| none (greedy) | — | 25.5% (26.5%) | 31.8% (32.9%) |
| in-domain 4-gram | 0.5 / 2.0 | 20.1% (21.0%) | 25.6% (26.7%) |
| **in-domain + external** | **0.75 / 3.0** | **15.5%** (16.2%) | **20.5%** (21.4%) |

The grid covered alpha 0.35–1.25 and beta 1–5; the optimum is interior.

**Caveat: dev validation flatters the external LM.** The validation sentences were sampled from
SODA *train* dialogues. The filter removes the sentences themselves, but the other turns of the
same dialogues stay in the LM text and give it unusually close context. The test sentences
come from SODA *test* dialogues, which are not in the LM text, so we expected a smaller gain on
test. It turned out almost as large.

Test, scored once with alpha 0.75, beta 3.0 (final LM: sessions 1–140 ×100 + external,
40,829 unigram types). Whisper WER, ours in brackets:

| Model | LM | Voiced | Silent | OOV recall (V / S) |
| --- | --- | --- | --- | --- |
| final_vs | none (greedy) | 18.1% (19.2%) | 29.4% (30.6%) | 1.1% / 0.4% |
| final_vs | in-domain | 15.1% (16.0%) | 24.2% (25.4%) | 0.4% / 0.0% |
| final_vs | **in-domain + external** | **11.1%** (11.9%) | **19.6%** (20.6%) | 4.0% / 1.8% |
| baseline_vs_dev | in-domain + external (dev LM) | 13.2% (14.1%) | 21.8% (23.0%) | 3.8% / 3.1% |
| final_vs seed 1 | in-domain + external | 11.1% (11.9%) | 18.8% (19.8%) | |
| final_vs seed 2 | in-domain + external | 11.3% (12.1%) | 18.8% (19.8%) | |
| **3 seeds, mean ± std** | in-domain + external | **11.2 ± 0.1%** | **19.1 ± 0.5%** | |
| 3 seeds, averaged posteriors | in-domain + external | 14.4% (15.4%) | 23.8% (25.0%) | ✗ worse |

**Averaging CTC posteriors across seeds does not work** (`ensemble_logprobs.py`): greedy WER
rises from 18.1 / 29.4 to 18.9 / 31.8. Independently trained CTC models emit each token as a
spike on a single frame, and the spikes fall on different frames in different models; averaging
flattens them until blank wins and words drop out. Seeds have to be combined at the word level
instead (ROVER, see [05_llm_combination](../05_llm_combination)).

### Leakage check on the test gain

The 8-gram filter cannot catch near-copies of short sentences, so we checked directly: **202
test sentences have a copy in the filtered external text that differs by exactly one word**
(listed in `/workspace/lm/external/test_near_copies.txt`). They are almost all short generic
lines with a different name ("hey alex how are you doing", "thanks laura i appreciate your
support"); 160 of them have 5–9 words. The gain does not come from them:

| Test subset | Mode | Greedy | In-domain LM | + external | Gain from external |
| --- | --- | --- | --- | --- | --- |
| all 1,000 | voiced | 18.1% | 15.1% | 11.1% | −4.0 |
| 798 without a near-copy | voiced | 19.1% | 16.0% | 11.9% | −4.1 |
| all 1,000 | silent | 29.3% | 24.2% | 19.6% | −4.6 |
| 798 without a near-copy | silent | 30.6% | 25.4% | 20.8% | −4.6 |

The gain is also flat across sentence length (−4.0 to −4.6 points for <8, 8–12 and 13+ words).
For a strict comparison, report the 798-sentence numbers alongside the full test set.
