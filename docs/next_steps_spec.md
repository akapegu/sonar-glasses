# Next steps: from 19.6% silent toward ~15%, then onto our own rig

This is a spec for Claude Code. It follows `docs/improvements_spec.md`
(phases A and B are done; see `experiments/04` and `05`).

Current best on test (Whisper-normalised): **11.1% voiced / 19.6% silent**
(final_vs, epoch 199, beam search with an external-text 4-gram LM).

**Timebox: about 2 weeks.** Prioritise changes that carry over to our own rig
and to new users. Skip changes that only help on this benchmark.

**Realistic target for the 2 weeks: ~16–17% silent** (stretch: 15%). Gains do not add up
linearly on top of a strong LM: the LM already fixes many of the errors the steps below target.

**Ceiling for selection-only methods.** The oracle WER of the current best system's top-10
beams (always picking the best of the ten) is **8.4% voiced / 15.8% silent**. Re-ranking or LLM
selection cannot go below that; reaching <10% silent needs a better acoustic model, more data
or better sensing (see step 0b).

**GPU budget.** A 40-epoch dev ablation costs about $1.50; a full 200-epoch final run about
$8.40 on 2× RTX 4090 (about 4.7 h). Steps 3–6 imply roughly 4–6 finals and 8–10 ablations:
**about $50–70**.

---

## 0. Rules (unchanged)

- Headline metric: Whisper-normalised WER, voiced and silent separately. Also
  report the strict 798-sentence test subset (no near-copy in the external text).
- Tune only on dev validation (sessions 132–140), using dev-split models
  (trained on 1–131). Score test once per finished method.
- Every experiment gets a folder under `experiments/NN_name/` with a README
  table: method, settings, dev validation, test, git hash.
- Change one thing at a time.
- Extra metrics on every evaluation:
  - WER by word position (first, middle, last).
  - OOV recall.
  - Perfect-sentence rate.
  - **"LM made it worse" rate:** the share of sentences with more errors than greedy.

---

## Step 0b. Data-scaling curve: does more data help with this sensing? (experiments/06_data_scaling)

Answers whether <10% silent is reachable by collecting more data, or whether the eyeglass
sensing is the bottleneck. Do this early: it informs the rig design.

1. Train the final recipe on 16, 33, 66 and 131 dev sessions (both modes, nested subsets),
   full 200-epoch schedule. Smaller subsets train proportionally faster (about $15 in total).
2. Score each on dev validation: greedy and with the external LM, voiced and silent separately.
3. Fit WER = a·N^(−b) + c per mode and decoder. Report b (how fast data helps), c (the floor
   more data never removes) and the sessions needed for 10% silent if c allows it.
4. Interpretation: c well below 10% → collect data (step 7.5). c near or above ~12% → the
   sensing is the bottleneck: more acoustic paths, wider bandwidth, extra sensors.

---

## Step 1. Fix LM over-trust (experiments/07_lm_retune)

Evidence: in a random 50-sentence sample, garbled silent sentences are turned
into fluent, unrelated lines ("then go you'll have a blast" → "hey carl you have
a minute"), and some sentences get worse than greedy. alpha 0.75 / beta 3.0 was
tuned on a dev set that the external LM flatters.

1. **Dialogue-level filtered dev LM.** Rebuild the dev external text, removing
   every SODA dialogue that contains any validation sentence (not just the
   sentence). Also run the same check for test sentences against SODA train,
   and log any matches.
2. **Retune alpha and beta** on dev validation with that LM. Grid alpha 0.2–1.25,
   beta 0–4.
3. **Confidence-adaptive LM weight.** Per utterance, compute the echo model's
   confidence (mean max posterior over non-blank frames, or greedy-path
   log-prob per frame). Lower alpha when confidence is low. Try 2–3 simple
   rules, for example two alpha values split at the median confidence. Tune on
   dev validation.
4. Score the best setting on test once.

Report: dev validation and test WER, the "LM made it worse" rate, and 10
example sentences before and after.

---

## Step 2. Check utterance ends (experiments/08_utterance_margins)

Evidence: final words are often dropped ("broken my arm" → "put my").

1. From existing results, compute WER for the last word and the last 3 words
   versus the middle. If the end is clearly worse, continue. Current numbers (silent, with
   in-domain LM): last word 27.6% vs middle 25.1%, so the effect is mild.
   Checked: the audio is not zeroed exactly at `end_s`; in the 0.2 s after it, 55–90% of
   samples are non-zero, so a margin does add real signal.
2. Check the cut: in preprocessing, the end frame comes from `end_s`. Plot the
   differential profile energy for the last 0.5 s of 50 utterances to see
   whether motion is still present at the cut.
3. Re-cut test (and dev validation) with a margin after `end_s` (try 0.2 s and
   0.4 s; never past the next utterance's start). Re-decode with the same
   models. A model trained on tight cuts may not use the margin until it is retrained with it,
   so if test-time margins show nothing, try one 40-epoch dev run with margins in training.
4. Also check whether beta (word bonus) interacts: a too-low beta favours short
   outputs.

---

## Step 3. Seeds, ensembling and safe LLM selection (experiments/09_seeds_rescoring)

1. Finish seeds 1 and 2 of final_vs. Report the mean and spread of test WER
   (greedy and with LM). This sets how big a gain counts as real.
2. **Seed ensemble (cheap, do first).** Average the three seeds' per-frame log-probabilities
   (they share the tokenizer), then decode with the LM as usual. Typically 1–2 points; no LLM.
   Tune nothing new: reuse alpha/beta from step 1.
3. **N-best rescoring instead of rewriting.** Candidates: the top-10 beams of
   each model in the pool, de-duplicated. Score each candidate as
   `CTC/LM beam score + λ × LLM log-likelihood` (length-normalised) using a base
   model (for example Qwen2.5-7B, not the instruct version). Pick the best. Tune λ on dev
   validation. This cannot invent words.
4. Compare on dev validation, then test, against: the best single model, the seed ensemble,
   ROVER, and the off-the-shelf LLM rewrite from experiment 05. Report WER and the share of
   outputs with a word in no candidate.
5. **Required before reporting any LLM result on test: a memorisation check.** SODA has been
   public on Hugging Face since 2022, so Qwen2.5 has very likely seen it, including SODA's
   test split, where our test sentences come from. Give the LLM the first 3 words of each test
   sentence and measure the exact-continuation rate against control sentences not in SODA
   (e.g. our own written sentences). If it is clearly higher, mark LLM test results as
   possibly contaminated.

---

## Step 4. Use the faulty-mic sessions (experiments/10_faulty_mic)

1. Add voiced 150–160 and silent 141–160 to training (their mic 1 is a copy
   of mic 0).
2. **Channel duplication augmentation:** per utterance, with p = 0.15, copy one
   mic's two paths (s0mX, s1mX) over the other mic's. Resample every epoch.
3. Dev: a 40-epoch ablation against the current recipe (note: dev validation
   sessions 132–140 must stay out of training). Final: a full 200-epoch run.

Expected: SoniGAN measured +1.4 points greedy; on top of our LM expect about 0.5–1. The
sessions are already in `cache/v1` (flagged `faulty_session`); only `src/data.py`'s row
selection and the augmentation need adding. The duplication trick also prepares the model
for a dead mic on our rig.

---

## Step 5. Better acoustic model (experiments/11_transformer_crosscon)

The biggest transferable gain. Do 5a, then 5b, each as its own ablation. Budget about
4 days for 5a + 5b. SoniGAN's 3.1-point crossCon gain was measured greedy; on top of our
LM expect less (about 1.5–3 points for 5a + 5b together).

**5a. ResNet-18 + Transformer**
- ResNet-18 front end (GroupNorm, no max-pool, 16× time downsampling, mean
  over range) feeding a 6-layer Transformer encoder: d_model 384, 8 heads,
  relative positional attention, dropout 0.1. Linear output to 1,001 classes,
  CTC loss.
- AdamW lr 3e-4, weight decay 0.01, cosine schedule with warm-up.
- Keep the current augmentations.

**5b. crossCon loss**
- **Prerequisite: the audible audio.** The training cache holds only echo profiles. Extract the
  audible band of the voiced takes (low-pass below 8 kHz, resample to 16 kHz) per utterance
  from the raw `audio.wav`: from the extracted copy on the Mac (`data/`, 48 GB) or by
  re-downloading the zips. Store them next to the cache.
- Targets: frame-level speech features from the **voiced** audio of the same
  sentence. Start with HuBERT-base layer 6 or 9 features, pooled to 12.5 Hz;
  soft speech units are an alternative.
- Voiced takes: use their own audio, time-aligned directly.
- Silent takes: align the voiced take to the silent take with DTW on
  echo-feature representations, then borrow the warped audio features.
- Loss: InfoNCE between projected ResNet outputs and projected speech
  features; a positive = same time step; negatives = other steps in the batch,
  from both modalities. Temperature 0.1 to start.
- Total loss = CTC + crossCon (weight 1.0 to start; tune on dev).

**5c. Auxiliary phoneme loss (only if time remains; moved out of the 2-week scope)**
- Forced-align voiced audio with the Montreal Forced Aligner to get phone
  boundaries. Map them to the 12.5 Hz output rate; carry them to silent takes
  through the same DTW.
- Add a phoneme classification head (39 ARPABET + silence) on the encoder
  output, used only in training. Total loss = CTC + crossCon + w × phoneme
  (w 0.3 to start).
- Montreal Forced Aligner setup plus the DTW carry-over is about 2–3 days. The large gain
  Gaddy reported was for EMG and is untested on echo profiles.

Report each against the current recipe at 40 epochs on dev, then a full run of
the best combination.

---

## Step 6. Rare and new words (experiments/12_vocab, only if time remains)

- Train a SentencePiece model with 500 units (and try 250); retrain at
  40 epochs; compare WER and OOV recall.
- Alternative: keep 1,000 units for the main head and add a character-level
  CTC head trained jointly (weight 0.3). Decode with the main head; report OOV
  recall for both.
- Expect little change in WER (smaller units often cost a little); the point is OOV recall,
  which is near zero today.

---

## Reaching <10% silent (beyond these 2 weeks)

Rough path, to be revised with the step-0b scaling fit:

| Stage | Silent (estimate) |
| --- | --- |
| Now: final recipe + external LM | 19.6% |
| This plan (ensemble, faulty sessions, Transformer + crossCon, rescoring) | ~16–17% |
| 3–5× data per user, mostly self-labelled voiced (step 7.5) | ~11–13%, if the floor c allows |
| Self-supervised pretraining on all-day unlabelled wear | ~1–2 points more |
| Product context: personal LM, conversation context, user-selectable suggestions | <10% in real use |

No published open-vocabulary silent-speech system from minimally obtrusive sensing is below
10% today (EMG with an LLM: 12.2%; SoniGAN: 14.9%). Product claims of 3–5% should be compared
only after checking the sensor, the vocabulary (closed or open), the context used and how WER
is scored. If step 0b shows a high floor, the lever is the hardware: more acoustic paths,
wider bandwidth for finer range resolution, or an extra sensor such as a jaw IMU.

---

## Skip for now

- SoniGAN's synthetic echo-profile generator: small gain, heavy work.
- Further SODA-specific LM tuning beyond step 1: does not carry over to real users.
- LoRA fine-tuning of the LLM: not enough examples yet.

---

## Step 7. Prepare for our rig (in parallel with steps 4–6)

These carry the work above onto real users.

1. **Rig preprocessing parity:** run our capture → PDM-to-PCM → echo profile
   pipeline and confirm the output format matches the SoniSpeech cache
   (4 × 80 × T at 200 fps, same normalisation path). Re-derive the crop window
   on our rig's data, since the direct path may sit elsewhere.
2. **Fine-tuning entry point:** load any checkpoint, optionally freeze layers,
   fine-tune on a small user dataset. Freeze options: none / all but the first
   block + output / output only.
3. **Calibration layer:** a learned shift-and-stretch on the range axis (and a
   per-channel gain) in front of the network, trained alone during
   calibration. Test it on SoniSpeech by treating each test session as a new day.
4. **Two-user experiment plan** (run once the rig works):
   - Record user A for 5–10 h and user B for 2–3 h, voiced and silent, with
     refits between sessions.
   - 200 shared test sentences that are never trained on.
   - 50 calibration sentences for B (about 5 min).
   - Experiments:
     1. A → A.
     2. A → B with no calibration.
     3. A + B calibration at 1, 3, 5 and 10 min → B (full fine-tune versus the calibration layer).
     4. A + B pooled → both.
     5. Refit: early sessions → a later session, per user.
   - **Key plot:** user B's WER against calibration minutes.
5. **Self-labelled voiced data (the scalable data source).** When the user speaks normally
   with the glasses on, the same microphones record the audible speech. Transcribe it with an
   ASR model (e.g. Whisper), keep segments above a confidence threshold, and use the
   transcripts to label the simultaneous echo profiles. This yields hours of labelled voiced
   data per day with no manual work; crossCon (step 5b) carries it over to silent speech.
   Plan the rig's recording format and privacy handling for this from the start.
6. **Unlabelled echo profiles for self-supervised pretraining.** All-day wear produces echo
   data with no transcripts at all; keep it for masked-prediction pretraining (phase D).
