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
