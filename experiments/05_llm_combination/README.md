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

LoRA on about 690 examples has not beaten zero-shot so far. Test results wait for the two
extra final-recipe seeds (test-time pool: final models + the dev models). Still to do: the
hallucination read-through of 50 changed sentences (B6), and test scoring.
