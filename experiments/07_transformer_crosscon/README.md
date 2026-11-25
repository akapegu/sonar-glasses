# 07 Better acoustic model: Transformer encoder (5a) and crossCon (5b)

Step 5 of [docs/next_steps_spec.md](../../docs/next_steps_spec.md). Every variant is a 40-epoch
dev run (train sessions 1–131, validate on 132–140, full 200-epoch schedule stopped after
40 epochs) compared at epoch 39 with the final recipe trained the same way
(`final_vs_dev_ref`, see [02_ablations](../02_ablations)).

```bash
bash experiments/02_ablations/ablate.sh experiments/03_final_recipe/configs/final_vs.yaml "ref|split=dev"
bash experiments/02_ablations/ablate.sh experiments/07_transformer_crosscon/configs/r18tf_vs.yaml "5a|train.seed=0"
```

## 5a. ResNet-18 + Transformer: worse, overfits

`configs/r18tf_vs.yaml`: ResNet-18 front end (GroupNorm, no max-pool, 16× time downsampling,
mean over range) → linear 512 → 384 → 6 pre-LayerNorm Transformer layers with rotary
self-attention (relative positions), 8 heads, feed-forward 1,536, dropout 0.1, padding masked →
1,001 classes, CTC. AdamW 3e-4, weight decay 0.01, cosine schedule with 5% warm-up. Same data,
augmentation and batch as the final recipe. 22.4 M parameters (ResNet-34: 21.8 M). 88 s per
epoch on an A100 SXM (ResNet-34: 107 s).

Dev validation, mean WER (our normalisation):

| Epoch | 4 | 9 | 14 | 19 | 24 | 29 | 34 | 39 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference (ResNet-34) | 95.7 | 61.9 | 47.4 | 41.2 | 38.8 | 37.3 | 37.2 | **35.1** |
| 5a (ResNet-18 + Transformer) | 95.0 | 81.9 | 51.8 | 46.6 | 45.3 | 45.3 | 45.3 | 43.4 |

| Epoch 39 | Voiced | Silent | Mean | Validation loss | Training loss |
| --- | --- | --- | --- | --- | --- |
| reference | 32.3% (Whisper 31.1%) | 37.8% (36.4%) | **35.1%** | 1.58 | 0.56 |
| 5a | 41.8% (40.3%) | 45.1% (43.3%) | 43.4% | 3.09 | 0.30 |

5a warms up more slowly, nearly catches up by epoch 14 (+4.4 points), then stalls from epoch 24
on while the reference keeps improving. Its training loss is half the reference's and its
validation loss twice, so it memorises the training sessions instead of generalising: with
about 13,000 training utterances, the attention layers overfit more than convolutions do.
Not worth a 200-epoch run as configured. More dropout or a smaller Transformer might narrow the
gap, but the larger expected gain is from 5b. (Test, for information: reference 25.5% / 37.5%,
5a 34.5% / 42.2%, Whisper, greedy.)

## 5b. crossCon: −4.9 points

`configs/crosscon_vs.yaml`: the final recipe (ResNet-34, aug3 + echoaug) unchanged, plus a
contrastive loss that pulls the encoder output at each CTC step towards speech features of the
same moment.

1. `extract_audio.py` cuts the audible speech of every voiced utterance (the two microphones
   averaged, 100 → 16 kHz with a polyphase low-pass, which removes the 18–39 kHz chirps), exactly
   over the samples its echo frames cover: 80 audio samples per echo frame. 8,976 utterances,
   1.4 GB, under a minute on the Mac.
2. `make_targets.py` runs HuBERT-base (`facebook/hubert-base-ls960`) layer 9 on that audio
   (50 Hz) and mean-pools 4 frames → 12.5 Hz, one 768-d vector per CTC output step. Silent takes
   borrow the vectors of a voiced take of the same sentence (6,535 of 6,540 have one), aligned with
   DTW on the encoder features of `baseline_vs_dev` (cosine distance; median path cost 0.19, 90th
   percentile 0.23). Validation and test utterances never get targets.
3. Training: `CrossConHead` projects the encoder output (512) and the HuBERT vector (768) to 256 d
   through a 2-layer MLP each; symmetric InfoNCE over all valid steps in the batch (positive =
   same step, negatives = every other step, both directions), temperature 0.1, up to 2,048 pairs
   per batch, weight 1.0, added to the CTC loss. The dataset tracks concatenation gaps and shift
   padding so each step gets the target of the frame at its centre (none in gaps and padding).
   The head is used in training only; decoding is unchanged.

```bash
.venv/bin/python experiments/07_transformer_crosscon/extract_audio.py --data data/SoniSpeech_0*_of_09 \
    --manifest /workspace/cache/v1/manifest.csv --out cache/audio16k     # then copy to /workspace/cache/audio16k
/workspace/venv/bin/python experiments/07_transformer_crosscon/make_targets.py --cache /workspace/cache/v1 \
    --audio /workspace/cache/audio16k --out /workspace/cache/crosscon_h9 --align-run /workspace/runs/baseline_vs_dev
bash experiments/02_ablations/ablate.sh experiments/07_transformer_crosscon/configs/crosscon_vs.yaml "cc|train.seed=0"
```

Dev validation, mean WER (our normalisation):

| Epoch | 4 | 9 | 14 | 19 | 24 | 29 | 34 | 39 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference | 95.7 | 61.9 | 47.4 | 41.2 | 38.8 | 37.3 | 37.2 | 35.1 |
| **+ crossCon** | 70.2 | 50.6 | 40.0 | 35.9 | 34.7 | 32.6 | 31.9 | **30.2** |

| Epoch 39 | Voiced | Silent | Mean | Validation loss |
| --- | --- | --- | --- | --- |
| reference | 32.3% (Whisper 31.1%) | 37.8% (36.4%) | 35.1% | 1.58 |
| **+ crossCon** | **26.7%** (25.7%) | **33.6%** (32.1%) | **30.2%** | **1.37** |
| change | −5.6 | −4.2 | **−4.9** | |

- crossCon is ahead at every check, and by the most early on (−25 points at epoch 4): the speech
  features give the encoder a dense, per-step training signal long before CTC alone has learned
  where words are.
- The gain holds for silent speech (−4.2), whose targets are borrowed through DTW from a
  different recording, not only for voiced takes with their own audio (−5.6).
- Validation loss is lower (1.37 vs 1.58) while training CTC loss is similar (0.44 vs 0.56), so
  this is better generalisation, not just faster fitting. The InfoNCE loss fell from 6.5 to 1.4.
- SoniGAN reported +3.1 points greedy from crossCon on a full run; here it is 4.9 at 40 epochs.
  The gap may shrink with 200 epochs, as the reference catches up late in training.
- Test, for information only (greedy, Whisper): reference 25.5% / 37.5%, crossCon 21.2% / 33.4%.
- Cost: 118 s per epoch on an A100 PCIe (reference 116 s); targets 1.8 GB on the volume.

Next: a full 200-epoch run of the final recipe + crossCon on sessions 1–140 (targets rebuilt with
`--split final`), then 3 seeds, the blank-penalty decoding and the reranker on top. That is the
path to a new headline; it waits for a go-ahead (about $25–30 on A100s for 3 seeds).
