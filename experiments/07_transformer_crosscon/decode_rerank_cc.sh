#!/usr/bin/env bash
# After the three final_cc seeds: blank-penalty LM decoding of each, then ROVER and the
# selection-only LLM reranker (7B and 3B) with the weights tuned on dev validation for the
# previous models (rerank_val_a0.75_b1.5_bp2.0*_weights.json), plus 2-fold session CV.
#   bash experiments/07_transformer_crosscon/decode_rerank_cc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
D=/workspace/venv-decode/bin/python; L=/root/venv-llm/bin/python; X=experiments
RUNS="final_cc_final final_cc_final_seed1 final_cc_final_seed2"
S=_w128_bp2.0
for r in $RUNS; do
  f=/workspace/runs/$r/logprobs_test_last_decode_final_ext_a0.75_b1.5$S.csv
  [ -f $f ] || $D $X/04_lm_decoding/lm_decode.py decode --logprobs /workspace/runs/$r/logprobs_test_last \
    --tokenizer /workspace/runs/$r/tokenizer.model --lm /workspace/lm/final_ext --alpha 0.75 --beta 1.5 \
    --blank-penalty 2.0 --cache /workspace/cache/v1 --procs 16 | tail -2 | sed "s|^|$r: |"
done
T=$(for r in $RUNS; do echo -n "/workspace/runs/$r/logprobs_test_last_decode_final_ext_a0.75_b1.5$S.csv "; done)
for m in 7B 3B; do
  W=/workspace/lisa/rerank_val_a0.75_b1.5_bp2.0$([ $m = 7B ] && echo "" || echo "_q$m")_weights.json
  echo "=== rerank with Qwen2.5-$m"
  $L $X/05_llm_combination/combine.py rerank --decodes $T --model /root/models/Qwen2.5-$m-Instruct \
    --weights $W --cv-folds 2 --out /workspace/lisa/rerank_test_cc_q$m
done
echo DECODE_RERANK_DONE
