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

Test results wait for the two extra final-recipe seeds (test-time pool: final models + the
dev models).
