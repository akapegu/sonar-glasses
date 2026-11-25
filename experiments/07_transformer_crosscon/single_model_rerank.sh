#!/usr/bin/env bash
# On-device setup: ONE echo model + its top-10 beams + Qwen2.5-3B (selection only).
set -euo pipefail
cd /workspace/sonar-glasses
python -m venv --system-site-packages /root/venv-llm && /root/venv-llm/bin/pip install -q -r requirements-llm.txt
HF_HUB_ENABLE_HF_TRANSFER=1 /root/venv-llm/bin/huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /root/models/Qwen2.5-3B-Instruct > /dev/null 2>&1
L=/root/venv-llm/bin/python; C=experiments/05_llm_combination/combine.py; Q=/root/models/Qwen2.5-3B-Instruct
echo "=== tune on dev val: baseline_vs_dev alone (its top-10)"
$L $C rerank --decodes /workspace/runs/baseline_vs_dev/logprobs_val_best_decode_dev_ext_a0.75_b1.5_w128_bp2.0.csv --model $Q --out /workspace/lisa/rerank_val_single_q3B
for r in final_cc_final final_cc_final_seed1 final_cc_final_seed2; do
  echo "=== test: $r alone"
  $L $C rerank --decodes /workspace/runs/$r/logprobs_test_last_decode_final_ext_a0.75_b1.5_w128_bp2.0.csv --model $Q \
    --weights /workspace/lisa/rerank_val_single_q3B_weights.json --cv-folds 2 --out /workspace/lisa/rerank_test_single_${r}_q3B
done
echo SINGLE_DONE
