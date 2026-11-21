# 02 Ablations of the paper's unknown settings

Short dev runs that change one setting each. Every run keeps the full 200-epoch LR schedule
but stops after 40 epochs (`train.stop_after`), so its validation curve lines up epoch for
epoch with `baseline_vs_dev`. `ablate.sh` runs a queue, scores each run on validation and test,
prints a summary and stops the pod.

```bash
bash experiments/02_ablations/ablate.sh experiments/01_reproduction/configs/baseline_vs.yaml \
    "aug3|aug.time_masks=10 aug.range_mask_max=16" \
    "echoaug|aug.concat_consecutive=true aug.concat_gap_max=200 aug.scale=0.05 aug.shift_p=0.5 aug.shift_max=400"
```

## Results (dev validation, mean of voiced and silent WER, our normalisation)

| Run | Change | Epoch | Run | Baseline | Verdict |
| --- | --- | --- | --- | --- | --- |
| `echoaug` | EchoSpeech augmentations: consecutive concatenation with pauses, ×U(0.95, 1.05) scaling, random time shift in still padding | 39 | **34.8%** | 36.0% | kept |
| `aug3` | 10 time masks, range masks up to 16 bins | 39 | **35.0%** | 36.0% | kept |
| `aug2` | 6 time masks, range masks up to 16 bins | 39 | 35.6% | 36.0% | superseded by aug3 |
| `rs1222` | keep full range resolution in the stem (range stride [1,2,2,2]) | 39 | 36.2% | 36.0% | no gain at 2× cost |
| `nonorm` | no per-channel normalisation (as the official code) | 19 | 40.9% | 40.2% | level, stopped |
| `aug4` | 6 time masks up to 80 frames, 3 range masks up to 24 bins | 29 | 40.8% | 38.8% | too aggressive, stopped |
| `c250` | crop bins 250:330 (from the direct path) | 19 | 41.9% | 40.2% | voiced +3.1, stopped |
| `c240` | crop bins 240:320 | 14 | 45.3% | 44.4% | worse, stopped |
| `wide160` | crop bins 210:370 (160 bins) | — | — | — | not run |

Test (epoch 39, Whisper / ours), for information only:

| Run | Voiced | Silent |
| --- | --- | --- |
| `echoaug` | 25.3% / 26.6% | 37.7% / 39.1% |
| `aug2` | 25.5% / 26.8% | 37.0% / 38.6% |
| `aug3` | 25.9% / 27.3% | 37.4% / 38.7% |
| `rs1222` | 26.5% / 27.9% | 38.9% / 40.3% |

Conclusions: the input pipeline (crop, normalisation, range stride) is not where the gap to the
paper comes from. More augmentation helps, mostly by reducing memorisation (validation loss
1.79 → 1.61–1.67). The crop tests used a wider cache (bins 210:370, `--crop 210 370` in
preprocessing, `crop: [a, b]` in the config); cropped back to 226:306 it is bit-identical to
the standard cache.
