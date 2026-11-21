# Improving on the SoniSpeech baseline: implementation spec

This is a spec for Claude Code. It builds on the reproduced baseline in this
repo (final_vs, epoch 199). Ideas come from SoniGAN (arXiv 2610.15671) and
MONA LISA (arXiv 2403.05583).

Work through the phases in order. Phase A needs no training, so start there.

---

## 0. Ground rules (apply to every phase)

- **Headline metric:** WER scored after Whisper's `EnglishTextNormalizer`
  (from the `openai-whisper` package), applied to both reference and
  hypothesis. SoniGAN scores this way. Always also report the repo's own
  normalisation, side by side.
- **Report voiced and silent separately.** Silent is the headline.
- **Tune only on the dev validation split** (voiced and silent sessions
  132–140), using models trained on the **dev split** (sessions 1–131).
  The final model (trained on 1–140) has seen sessions 132–140, so its
  validation outputs must never be used for tuning.
- **Test is scored once per finished method**, with settings frozen from dev.
- **Leakage guard:** no test sentence, and no close match, may appear in any
  external text used for a language model or LLM fine-tuning (section A2).
- Log every run: method, settings, git hash, dev-val WER, test WER.
- Extra metrics on every evaluation:
  - OOV recall: the share of the 242 test-only word types that are output correctly.
  - WER by word position: first word, first three words, middle, last.
  - Perfect-sentence rate and the share of sentences that are more than half wrong.

Baseline rows to beat (Whisper-normalised, final_vs):

| Model | Voiced | Silent |
| --- | --- | --- |
| final_vs, greedy (epoch 199) | 18.1% (ours 19.2%) | 29.4% (ours 30.6%) |
| Paper (scoring not stated) | 15.8% | 26.3% |
| SoniGAN full system | n/a | 14.9% |

---

## Phase A: better decoding (no training)

### A1. Save CTC outputs

- Write a script that runs a checkpoint over a split and saves the per-utterance
  log-probabilities `(T', 1001)` plus the reference text, so decoding can be
  re-run without the GPU.
- Do this for: the final_vs checkpoint on test, and the dev baseline
  checkpoint (baseline_vs_dev, best) on dev validation and on test.

### A2. Build the language-model text

1. **In-domain:** normalised training transcripts from sessions 1–140
   (about 8,000 sentences). For the dev-tuning LM, use sessions 1–131 only.
2. **External conversational text** (optional, add after 1 works): the SODA
   *train* split, Cornell Movie Dialogs, OpenSubtitles. Apply the same
   normalisation the corpus uses: lowercase, strip punctuation, spell out
   numbers and currency.
3. **Leakage filter (required for 2):** remove any external sentence that
   - exactly matches a test or validation sentence after normalisation, or
   - shares any 8-word sequence with a test or validation sentence.

   Log how many sentences were removed.
4. Keep both LMs: in-domain only, and in-domain plus external.

### A3. N-gram LM and lexicon

- Build word 4-gram LMs with KenLM (`lmplz -o 4`, then `build_binary`).
- Lexicon (unigram list) = every word in the LM text.
- **Note:** with the in-domain-only LM, the 242 test-only words cannot be
  output. Measure OOV recall to see what that costs, and compare with the
  external LM.

### A4. Beam search

- Use `pyctcdecode` with the KenLM model. The label list must match the model:
  index 0 = blank (`""`), index k = SentencePiece piece k−1, keeping the `▁`
  word-start markers so the decoder can rebuild words.
- Parameters to tune on dev validation: `alpha` (LM weight), `beta` (word
  insertion bonus), `beam_width`. Start with alpha 0.5, beta 1.0, beam 128;
  grid alpha 0.2–1.5 and beta 0–3; try beam 128 and 512.
- Also save the top N = 10 beam hypotheses with scores, for phase B.

### A5. Report

| Decoder | LM text | Voiced | Silent | OOV recall |
| --- | --- | --- | --- | --- |
| Greedy | none | | | |
| Beam | in-domain | | | |
| Beam | in-domain + external | | | |

**Success criterion:** silent test WER below 26.3% with an LM. State clearly
that the paper's number has no LM.

---

## Phase B: LLM picks the sentence (LISA)

MONA LISA's biggest gains came from an LLM choosing among the top-1 guesses of
*several different models*, then fine-tuning that LLM on a small set.

### B1. Get diverse models

Diversity matters more than the count. Use what exists first:

- the final_vs model, and the dev baseline model;
- the dev ablation runs (aug3, echoaug, crop variants), if their checkpoints exist.

Then train two more seeds of final_vs (`train.seed=1`, `train.seed=2`). These
also give the seed spread for the baseline.

For tuning, only use models trained on the dev split, since they never saw
sessions 132–140.

### B2. Candidates

- For each utterance: the top-1 output of each model (phase A beam + LM
  decoding), plus the top 10 beams of the best model. Remove duplicates.

### B3. Simple voting baseline first

- ROVER-style word-level voting across the models' top-1 outputs (align with
  edit distance, take the majority word). This is the non-LLM baseline the
  LLM must beat.

### B4. Off-the-shelf LLM

- Model: a local open LLM (for example Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct)
  run with vLLM or transformers. No paid API is needed.
- Prompt: a short task description plus the numbered candidates. The LLM
  answers with one corrected sentence and nothing else.
- Rule in the prompt: prefer words that appear in the candidates; fix only
  what is clearly wrong.

### B5. Fine-tuned LLM

- Training data: dev validation sessions 132–140 (about 900 utterances across
  both modes). Input = candidates from dev-split models; target = reference.
- Hold out sessions 139–140 to measure, and train on 132–138.
- LoRA fine-tune (rank 16, a few epochs). Stop on the held-out WER.
- At test time the candidates come from the final models.

### B6. Hallucination checks

- Count outputs containing a word that is in none of the candidates.
- Compare OOV recall with phase A.
- Read 50 random changed sentences by hand and note any that read fluently but
  change the meaning.

### B7. Report

| Method | Voiced | Silent |
| --- | --- | --- |
| Best phase A | | |
| ROVER vote | | |
| LLM, off the shelf | | |
| LLM, fine-tuned | | |

---

## Phase C: better training (after A and B)

Change one thing per run. Use dev 40-epoch ablations to choose, as before,
then a full final run.

1. **Use the faulty-mic sessions** (voiced 150–160, silent 141–160) with
   **channel duplication**: per utterance, with p = 0.15, copy one mic's two
   paths over the other mic's two. Resample every epoch. SoniGAN: +1.4 points.
2. **ResNet-18 + Transformer recognizer:** 6 layers, d_model 384, 8 heads,
   relative position attention, same input and 16× downsampling. AdamW
   lr 3e-4, weight decay 0.01, cosine schedule.
3. **crossCon loss:** an InfoNCE loss between ResNet features and soft speech
   units from the voiced audio (e.g. HuBERT features pooled to 12.5 Hz),
   with negatives from both modalities. Silent takes borrow units from the
   voiced take of the same sentence, aligned with DTW on echo features.
   Total loss = CTC + crossCon. SoniGAN: worth 3.1 points.
4. **Auxiliary phoneme loss:** forced-align voiced audio (Montreal Forced
   Aligner), carry phoneme labels to silent takes by the same DTW, and add a
   frame-level phoneme classification head during training only. Gaddy 2021
   (EMG) found this the largest single gain.

---

## Phase D: later research bets

- A shared audio–echo latent space trained with LibriSpeech audio (MONA).
- Aligning echo features to Whisper encoder outputs and reusing Whisper's
  decoder (Aleph's approach).
- A lip-reading model on the frontal video as a teacher (untried on SoniSpeech).
- Self-supervised pretraining on echo profiles.
- A small learned shift-and-stretch layer on the range axis, for refit and
  new-user calibration on our own rig.

---

## Order and timing

| When | Work |
| --- | --- |
| Before Nov 29 | Phase A complete; phase B4 if time allows |
| Early December | B1–B7 (needs 2 extra seeds) |
| December | Phase C, one change at a time |
| Later | Phase D |

Hardware bring-up takes priority over everything here until the rig works.
