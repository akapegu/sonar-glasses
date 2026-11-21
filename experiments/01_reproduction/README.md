# 01 Reproduction: the paper's recipe

ResNet-34 (GroupNorm 32, no max-pool, 16× time downsampling) + CTC over 1,000 SentencePiece
pieces; Adam lr 2e-4, weight decay 1e-4, 5% warm-up then exponential decay, 200 epochs,
batch 16; SpecAugment (2 range, 3 time masks) and utterance concatenation. Settings the paper
does not give are guesses, listed in [docs/reproduction_spec.md](../../docs/reproduction_spec.md) section 10.

| Config | Training data | Paper (test, V / S) |
| --- | --- | --- |
| `configs/baseline_vs.yaml` | voiced + silent | 15.8% / 26.3% |
| `configs/baseline_v.yaml` | voiced only | 16.9% / 78.4% |
| `configs/baseline_s.yaml` | silent only | 55.7% / 33.7% |

All three default to the dev split (train 1–131, validate on 132–140). `--set split=final`
trains on 1–140 with no validation set, as in the paper.

```bash
bash scripts/pod_pipeline.sh experiments/01_reproduction/configs/baseline_vs.yaml   # on a pod
python scripts/train.py --config experiments/01_reproduction/configs/baseline_vs.yaml   # anywhere
```

## Results: `baseline_vs_dev` (sessions 1–131, 200 epochs, best checkpoint = epoch 199)

| Split | Voiced, Whisper (ours) | Silent, Whisper (ours) |
| --- | --- | --- |
| Dev validation | 25.5% (26.5%) | 31.8% (32.9%) |
| Test | 21.2% (22.3%) | 32.5% (34.0%) |

Gap to the paper on test (Whisper): +5.4 voiced, +6.2 silent. What we ruled out as causes:
pipeline bugs (a 20-utterance overfit test reaches 0% WER; preprocessing matches the official
code to 4e-4), decoding (the paper is greedy, no LM too), scoring (normalisation changes WER by
about 1–1.5 points), range downsampling and the crop window (see
[02_ablations](../02_ablations)). The chosen recipe and the 140-session run are in
[03_final_recipe](../03_final_recipe). The voiced-only and silent-only configs have not been run.

Independent check: SoniGAN (same lab) reports that its own ResNet-34, retrained on the same
140 sessions, reads real silent test profiles at 30.4% WER, so the published 26.3% is not
reached by the lab's own retrained baseline either.
