# sonar-glasses: silent speech from ultrasonic echoes

Reproducing and then improving the SoniSpeech baseline (ResNet-34 + CTC, arXiv 2608.00803)
on the public SoniSpeech dataset: eyeglasses with two speakers and two microphones that send
inaudible FMCW chirps (18–39 kHz) and read lip and jaw movement from the echoes.

**Status (2026-11-25).** The baseline is reproduced to within about 2.5–3 points of the paper
(greedy decoding, no language model; three seeds agree to 0.1 points). On top of it: a 4-gram
language model trained on the transcripts plus leakage-filtered conversational text, a blank
penalty in the decoder, and an LLM that picks one candidate from three seeds' outputs (it
never rewrites, so it cannot invent words). Together they take test WER to **8.5% voiced /
15.3% silent**.

## Headline results (test set, 1,000 sentences per mode)

WER after Whisper's `EnglishTextNormalizer`, the scoring SoniGAN states it uses on this test
set. Our own stricter normalisation keeps contractions and is about 1–1.5 points higher. Both
are in every experiment README.

| System | Voiced | Silent | Where |
| --- | --- | --- | --- |
| Paper: SoniSpeech baseline, greedy, no LM (scoring not stated) | 15.8% | 26.3% | — |
| SoniGAN's retrained ResNet-34 (200 test sentences) | — | 30.4% | — |
| Ours, dev baseline (paper recipe, sessions 1–131), greedy | 21.2% | 32.5% | [01_reproduction](experiments/01_reproduction) |
| **Ours, final recipe (sessions 1–140), greedy, mean of 3 seeds** | **18.1%** | **29.4%** | [03_final_recipe](experiments/03_final_recipe) |
| Ours, final recipe + in-domain 4-gram LM | 15.1% | 24.2% | [04_lm_decoding](experiments/04_lm_decoding) |
| Ours, final recipe + 4-gram LM with external conversational text | 11.1% | 19.6% | [04_lm_decoding](experiments/04_lm_decoding) |
| Ours, ROVER vote over 3 seeds (each + external-text LM) | 10.2% | 18.0% | [05_llm_combination](experiments/05_llm_combination) |
| Ours, final recipe + external-text LM + blank penalty (mean of 3 seeds) | 10.4% | 18.1% | [04_lm_decoding](experiments/04_lm_decoding) |
| **Ours, 3 seeds (LM + blank penalty), LLM picks one candidate** | **8.5%** | **15.3%** | [05_llm_combination](experiments/05_llm_combination) |
| SoniGAN full system (synthetic data, crossCon, 160 sessions, LM, LLM, ensemble) | — | 14.9% | — |

On the strict test subset (798 sentences with no near-copy in the LM's external text) the
best system reads 9.0% voiced / 16.2% silent. Every setting (decoder weights, blank penalty,
reranker weights) was chosen on dev validation; a 2-fold cross-validation over test sessions
gives the same result within 0.1 points. Letting the LLM rewrite the candidates instead
(Qwen2.5-7B) ties voting at 18.0% silent but invents words in about 12% of outputs.
Seed spread of the greedy result: about 0.1 points. The comparison with the paper is not like
for like: the paper decodes greedily without an LM.

## Layout

```
README.md                  this file
docs/
  reproduction_spec.md     the original spec for reproducing the paper (data facts, splits, recipe)
  improvements_spec.md     the spec for going beyond the paper (phases A-D)
  next_steps_spec.md       the 2-week plan toward ~15-17% silent, the data-scaling test, the rig
  runpod.md                how the cloud runs work (pods, volume, pipeline, gotchas)
src/                       library code shared by everything
  common.py                constants, splits, drop rules, text normalisation
  official.py              wrapper around the dataset's official echo-profile code
  data.py                  dataset, augmentation (SpecAugment, concatenation, EchoSpeech), batching
  model.py                 ResNet-34 + CTC
  decode.py                SentencePiece tokenizer, greedy CTC decoding, WER
  metrics.py               Whisper/our WER, OOV recall, position errors, sentence rates
  tracking.py              MLflow / W&B / CSV logging
  whisper_normalizers/     Whisper's text normaliser (vendored, MIT)
scripts/                   the shared pipeline, used by every experiment
  fetch_and_preprocess.py  download zips from eCommons and build the training cache
  preprocess.py            audio -> cropped differential echo profiles (float16, per utterance)
  train.py                 training (single GPU, or multi-GPU with torchrun)
  evaluate.py              greedy decoding + WER per mode (ours and Whisper)
  dump_logprobs.py         save per-utterance CTC log-probs for offline decoding
  pod_pipeline.sh          unattended pod run: train, evaluate, dump log-probs, stop the pod
experiments/
  00_data_analysis/        EDA reports, corpus checks, drop list, data explainer
  01_reproduction/         the paper's recipe (configs for voiced+silent, voiced-only, silent-only)
  02_ablations/            40-epoch ablations of unknown settings (crop, masks, normalisation, ...)
  03_final_recipe/         the chosen recipe trained on sessions 1-140, plus extra seeds
  04_lm_decoding/          phase A: n-gram LM beam search, external text, leakage filter
  05_llm_combination/      phase B: ROVER voting, LLM correction and reranking, LoRA fine-tuning
  06_phoneme_analysis/     phoneme error rate, per-sound recall, confusions, where deletions happen
data/ cache/ runs/         local data, preprocessing cache and run outputs (gitignored)
```

Each experiment folder has a README with what was run, the exact commands and the results.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# build the cache (downloads the 9 zips, ~90 GB transient, 13 GB cache)
.venv/bin/python scripts/fetch_and_preprocess.py --cache cache/v1 --work /tmp/sonispeech
# train the final recipe (2 GPUs; drop torchrun for one GPU)
torchrun --standalone --nproc_per_node=2 scripts/train.py \
    --config experiments/03_final_recipe/configs/final_vs.yaml --set cache=cache/v1 runs_dir=runs
# score the last checkpoint on test
.venv/bin/python scripts/evaluate.py --run runs/final_vs_final --ckpt last --split test
```

On RunPod the same steps run unattended through `scripts/pod_pipeline.sh`; see
[docs/runpod.md](docs/runpod.md). Decoding with a language model and the LLM step use their own
environments (`requirements-decode.txt`, `requirements-llm.txt`).

## Rules this repo follows

- Settings are chosen on dev validation (sessions 132–140, models trained on 1–131). Test is
  scored once per finished method; test numbers mid-training are informational only.
- No test sentence, or text sharing an 8-word sequence with one, enters any LM or LLM training
  text (`experiments/04_lm_decoding/build_external_text.py`).
- Voiced and silent are always reported separately, under both normalisations.
