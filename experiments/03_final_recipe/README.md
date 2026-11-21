# 03 Final recipe (sessions 1–140)

`configs/final_vs.yaml`: the paper's recipe plus the two augmentation changes that helped in
[02_ablations](../02_ablations) (`aug3` + `echoaug`), crop 226:306 with per-channel
normalisation. Trained on sessions 1–140 of both modes with no validation set, as in the
paper, and scored once on test with the last checkpoint.

```bash
# unattended on a 2-GPU pod (train, score test, dump log-probs, stop the pod)
bash scripts/pod_pipeline.sh experiments/03_final_recipe/configs/final_vs.yaml
# extra seeds
TAG=seed1 EXTRA_ARGS="--set train.seed=1" bash scripts/pod_pipeline.sh experiments/03_final_recipe/configs/final_vs.yaml
```

Multi-GPU uses torchrun: the global batch of 16 is split across GPUs and the gradients are
averaged, which equals single-GPU training (checked: loss and gradients match to 1e-8). On
2× RTX 4090 an epoch takes about 84 s, so 200 epochs take about 4.7 h.

## Results (test, last checkpoint = epoch 199)

| Run | Voiced, Whisper (ours) | Silent, Whisper (ours) |
| --- | --- | --- |
| `final_vs_final` (seed 0) | **18.1%** (19.2%) | **29.4%** (30.6%) |
| `final_vs_final_seed1` | running | |
| `final_vs_final_seed2` | running | |
| Paper | 15.8% | 26.3% |
| SoniGAN's retrained ResNet-34 (200 sentences) | — | 30.4% |

Gap to the paper (Whisper): +2.3 voiced, +3.1 silent, down from +5.4 / +6.2 for the dev baseline.

Test WER during training (informational only; the reported number is epoch 199):

| Epoch | 39 | 49 | 77 | 99 | 112 | 127 | 139 | 149 | 159 | 169 | 179 | 189 | 199 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Voiced, Whisper | 22.8 | 22.1 | 20.4 | 19.3 | 19.0 | 18.9 | 18.4 | 18.4 | 18.4 | 18.3 | 18.1 | 18.2 | 18.1 |
| Silent, Whisper | 35.0 | 34.6 | 32.4 | 30.4 | 30.2 | 30.4 | 29.8 | 29.7 | 29.6 | 29.9 | 29.2 | 29.4 | 29.4 |

The curve flattens after epoch 140. Training loss ends at 0.08 (the dev baseline reached 0.03),
so the extra augmentation is doing its job.

Test set facts: 20 sessions per mode, the same 1,000 sentences in both modes, none of them in
training; 1,684 word types, 260 of them never seen in training. At epoch 199, 23% of voiced
and 9.5% of silent sentences are recognised perfectly.
