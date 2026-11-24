# 05 LLM combination (improvements phase B, after MONA LISA)

Candidates per utterance: the LM-decoded top-1 of every model in a pool plus the top-10 beams
of the strongest model, de-duplicated. Then:

- **ROVER** (B3): word-level majority vote over the models' top-1 outputs, aligned on the
  strongest model's output.
- **LLM, off the shelf** (B4): Qwen2.5-7B-Instruct reads the numbered candidates and writes one
  corrected sentence (prompt in `combine.py`; prefer words from the candidates).
- **LLM, LoRA fine-tuned** (B5): rank 16 on dev validation sessions 132–138, early-stopped on
  139–140.

```bash
L=/root/venv-llm/bin/python; X=experiments/05_llm_combination
$L $X/combine.py rover --decodes A_decode.csv B_decode.csv ... --out /workspace/lisa/rover_val
$L $X/combine.py llm --decodes ... --model /root/models/Qwen2.5-7B-Instruct --out /workspace/lisa/llm0_val
$L $X/combine.py finetune --decodes ... --model /root/models/Qwen2.5-7B-Instruct --adapter-out /workspace/lisa/adapter
```

Environment: `requirements-llm.txt` (transformers + peft on the pod's own torch). The 7B model
needs about 15 GB of GPU memory to run and fits LoRA training on 32 GB with gradient
checkpointing.

## Results so far: dev validation (Whisper WER, ours in brackets)

Pool: the five dev-split models (`baseline_vs_dev` first, then `echoaug`, `aug3`, `aug2`,
`rs1222`), all decoded with the dev 4-gram LM.

| Method | Voiced | Silent | Outputs with a word in no candidate (V / S) |
| --- | --- | --- | --- |
| Best single model, beam + LM | 20.1% (21.0%) | 25.6% (26.7%) | 0% / 0% |
| ROVER vote | 19.8% (20.7%) | 25.5% (26.4%) | 0% / 0% |
| **Qwen2.5-7B, off the shelf** | **17.4%** (18.3%) | **22.8%** (23.9%) | 14.6% / 14.9% |

Held-out sessions 139–140 only:

| Method | Voiced | Silent |
| --- | --- | --- |
| Qwen2.5-7B, off the shelf (epoch 0) | 16.5% | 21.7% |
| LoRA, epoch 1 | 18.1% | 27.9% |
| LoRA, epoch 2 | 16.1% | 25.6% |
| LoRA, epoch 3 | 17.9% | 26.9% |
| LoRA, epoch 4 | 16.7% | 24.7% |

**LoRA does not beat zero-shot** (rank 16, lr 1e-4, effective batch 8, 4 epochs, 15 min on an
RTX PRO 4500). After fine-tuning, 62–65% of its outputs contain a word that is in none of the
candidates: with only about 690 examples it learns to write fluent sentences of its own rather
than to repair the candidates. The saved adapter is the epoch-0 (zero-shot) model.

## Hallucination check (B6, off-the-shelf Qwen on dev validation)

| | Utterances |
| --- | --- |
| Output differs from the strongest model's top-1 | 580 of 882 |
| ... fewer errors / more errors / same | 326 / 82 / 172 |
| Output contains a word in no candidate | 130 |
| ... fewer errors / more errors | 63 / 22 |

A hand read of 50 random new-word outputs: most repair word forms and grammar
("follow and too please" → "falling into place", 5 errors → 0). About 7 of the 50 read
fluently but change the meaning, almost always where the candidates were already garbled:
"she was movie" → "she was perfect" (reference: "moving"), "how in the i start" → "how it
started" (reference: "how did the fire start"). When the candidates are nonsense, the LLM writes
plausible text instead of flagging uncertainty.

## Memorisation check (required before reporting LLM results on test)

`memorisation_check.py` gives Qwen2.5-7B the first 3 words of a sentence and counts exact
5-word continuations: **test 0.24%** (2 of 849), SoniSpeech training sentences 0.10%
(1 of 1,000), Cornell control 0.00% (0 of 1,000). No sign that the LLM has memorised the test
sentences, although SODA is public.

## Results on test (scored once)

Pool: the three final-recipe seeds, each decoded with the external-text LM (alpha 0.75,
beta 3.0), plus seed 0's top-10 beams. Whisper WER, ours in brackets; "strict" = the 798 test
sentences with no one-word-different copy in the LM text.

| Method | Voiced | Silent | Strict V / S | Outputs with a word in no candidate (V / S) |
| --- | --- | --- | --- | --- |
| Seed 0 + external LM | 11.1% (11.9%) | 19.6% (20.6%) | 11.9% / 20.8% | 0% / 0% |
| **ROVER over 3 seeds** | **10.2%** (10.9%) | **18.0%** (18.8%) | **10.9% / 19.1%** | 0% / 0% |
| Qwen2.5-7B zero-shot over 3 seeds | 10.5% (11.4%) | 18.0% (19.0%) | 10.9% / 18.8% | 12.8% / 11.4% |

With three equally strong models, plain word voting matches the LLM on silent and beats it on
voiced, without inventing words. On dev validation the LLM gained more (−2.7 points) because
that pool had one strong model and four weak ones, which voting cannot exploit. ROVER over
the seeds is the best method so far.

## Selection-only reranking (`combine.py rerank`)

The LLM rewrite above invents words in about 12% of outputs, and LoRA made that worse. Here the
LLM never writes text: it only scores candidates, and the output is always one of them. Per
utterance, every candidate (the pool's top-1 outputs plus the strongest model's top-10 beams)
gets four features, and the highest weighted sum wins:

| Feature | What it is |
| --- | --- |
| `llm` | Qwen2.5-7B log-probability of the candidate as plain text (after a fixed one-line prefix, including a closing newline) |
| `agree` | minus the mean word edit distance to the pool's top-1 outputs (a consensus / minimum-Bayes-risk term) |
| `dec` | the strongest model's beam score (LM included); candidates not in its top 10 get its 10th score |
| `nw` | number of words (a length bonus) |

The weights are found by grid search (2,205 combinations of the four) on dev validation, with
the dev pool (`baseline_vs_dev` plus the four 40-epoch ablation models, each decoded with the dev
external LM), and then frozen for test. As a second check, `--cv-folds 2` tunes on half the test
sessions and scores the other half, both ways (the sentence sets of the two halves are
disjoint). The best weights with `llm` fixed at 0 are also reported, to show what the LLM adds.
Scoring 16,500 candidates takes about 5 minutes on an A100.

```bash
L=/root/venv-llm/bin/python; X=experiments/05_llm_combination; Q=/root/models/Qwen2.5-7B-Instruct
$L $X/combine.py rerank --decodes <5 dev-pool val decodes> --model $Q --out /workspace/lisa/rerank_val_a0.75_b1.5_bp2.0
$L $X/combine.py rerank --decodes <3 seed test decodes> --model $Q --cv-folds 2 \
    --weights /workspace/lisa/rerank_val_a0.75_b1.5_bp2.0_weights.json --out /workspace/lisa/rerank_test_a0.75_b1.5_bp2.0
```

Weights tuned on dev validation for the blank-penalty decodes: with LLM `llm 0.75, agree 4,
dec 0.5, nw 4`; without `agree 16, dec 0.1, nw 3`. Dev validation itself (in-sample, so
optimistic): top-1 14.9 / 19.9, ROVER 13.9 / 18.3, rerank without LLM 14.1 / 18.7, **with LLM
12.8 / 17.0**.

### Results on test (scored once)

Pool: the three final-recipe seeds plus seed 0's top-10 beams. Whisper WER, ours in brackets;
strict = the 798 sentences with no one-word-different copy in the LM text. Every method outputs
one of the candidates, so no output contains a new word.

| Decoder setting | Method | Voiced | Silent | Strict V / S | Perfect silent sentences |
| --- | --- | --- | --- | --- | --- |
| b 3.0, no penalty | seed 0 alone | 11.13% (11.92%) | 19.63% (20.62%) | 11.9 / 20.8 | 25.8% |
| b 3.0, no penalty | ROVER | 10.23% (10.93%) | 17.98% (18.84%) | 10.9 / 19.1 | 29.1% |
| b 3.0, no penalty | rerank, no LLM | 10.10% (10.83%) | 17.56% (18.61%) | | 29.1% |
| b 3.0, no penalty | rerank + LLM | 9.94% (10.65%) | 17.42% (18.33%) | 10.5 / 18.4 | 28.5% |
| penalty 2.0, b 1.5 | seed 0 alone | 10.28% (10.95%) | 18.21% (19.14%) | 11.0 / 19.3 | 28.9% |
| penalty 2.0, b 1.5 | ROVER | 9.51% (10.15%) | 16.73% (17.61%) | 10.2 / 17.8 | 32.0% |
| penalty 2.0, b 1.5 | rerank, no LLM | 9.55% (10.16%) | 16.48% (17.35%) | 10.3 / 17.5 | 31.7% |
| penalty 2.0, b 1.5 | **rerank + LLM** | **8.48%** (9.08%) | **15.29%** (16.23%) | **9.0 / 16.2** | **35.2%** |
| penalty 2.0, b 1.5 | rerank + LLM, 2-fold session CV | 8.60% (9.19%) | 15.35% (16.27%) | | 34.8% |

- **New best: 8.5% voiced / 15.3% silent** (strict 9.0 / 16.2), down from ROVER's 10.2 / 18.0.
  The blank penalty and the reranker add up: the penalty gives better candidates, the LLM picks
  among them better than voting.
- The LLM is worth about 1 point on top of consensus selection (silent 16.5 → 15.3, voiced
  9.6 → 8.5). Without the LLM, selection is about level with ROVER.
- Tuning on dev and the session cross-validation agree within 0.1 points, so the weights
  transfer from the dev pool (one strong model, four weak ones) to three equally strong seeds.
- The gain is the same size on the strict subset (−1.9 voiced, −2.9 silent against ROVER), so it
  does not come from test sentences with near-copies in the LM text. The memorisation check
  above found no sign that Qwen has memorised the test sentences.
- Silent 15.3% is below the old selection ceiling (15.8%: the best of seed 0's top 10 at the old
  setting), because the pool now holds three seeds and the penalty decodes are better.

### How small can the reranking LLM be?

Same pipeline with smaller Qwen2.5 Instruct models (blank-penalty decodes, 3 seeds; weights
re-tuned on dev validation for each model, then frozen; `--model /root/models/Qwen2.5-<size>-Instruct`).
Test, Whisper WER (ours in brackets):

| Reranker LLM | Voiced | Silent | Strict V / S | 2-fold session CV (V / S) | Share of the 7B gain kept (V / S) |
| --- | --- | --- | --- | --- | --- |
| none (consensus selection only) | 9.55% (10.16%) | 16.48% (17.35%) | 10.3 / 17.5 | 9.51 / 16.53 | — |
| Qwen2.5-0.5B | 8.75% (9.33%) | 15.62% (16.50%) | 9.3 / 16.6 | 8.90 / 15.65 | 75% / 72% |
| Qwen2.5-1.5B | 8.70% (9.32%) | 15.43% (16.36%) | 9.3 / 16.4 | 8.79 / 15.48 | 79% / 88% |
| **Qwen2.5-3B** | **8.52%** (9.12%) | **15.41%** (16.31%) | **9.1 / 16.4** | 8.67 / 15.48 | **96% / 90%** |
| Qwen2.5-7B | 8.48% (9.08%) | 15.29% (16.23%) | 9.0 / 16.2 | 8.60 / 15.35 | 100% |

- **A 3B model keeps 90–96% of the 7B's gain; even 0.5B keeps about three quarters.** The
  differences between 1.5B, 3B and 7B (0.1–0.2 points) are close to the noise between the
  dev-tuned and cross-validated results.
- Scoring is one forward pass per candidate (about 8 candidates per utterance) with no
  generation, so a 3B model quantised to 4 bits (about 2 GB) can run on a phone. The whole
  correction step can stay on the device; no text has to leave it.
- Smaller models get a smaller tuned LLM weight relative to the other features (3B: llm 0.2,
  agree 1, dec 0.1, nw 1; 7B: llm 0.75, agree 4, dec 0.5, nw 4); the ratios are similar, so the
  LLM's share of the decision is about the same.
