#!/usr/bin/env bash
# One seed of the final recipe + crossCon on a pod: build the targets (lead pod only; the
# others wait for them), then the standard unattended pipeline (train, score test, dump
# log-probs, stop the pod).
#   tmux new-session -d -s cc "bash experiments/07_transformer_crosscon/run_final_cc.sh 0 lead"
#   tmux new-session -d -s cc "bash experiments/07_transformer_crosscon/run_final_cc.sh 1"
set -uo pipefail
SEED=$1; ROLE=${2:-follow}
cd "$(dirname "$0")/../.."
OUT=/workspace/cache/crosscon_h9_final
stamp() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }
if [ "$ROLE" = lead ]; then
  if [ ! -f $OUT/targets.csv ]; then
    echo "[$(stamp)] building crossCon targets for the final split"
    /workspace/venv/bin/python experiments/07_transformer_crosscon/make_targets.py --split final \
      --cache /workspace/cache/v1 --audio /workspace/cache/audio16k --out $OUT \
      --align-run /workspace/runs/baseline_vs_dev --layer 9 || { echo "[$(stamp)] targets failed"; exit 1; }
  fi
else
  echo "[$(stamp)] waiting for $OUT/targets.csv"
  until [ -f $OUT/targets.csv ]; do sleep 30; done
  sleep $((SEED * 45))  # stagger the pipelines' git pulls on the shared checkout
fi
TAG=$([ "$SEED" = 0 ] && echo "" || echo "seed$SEED")
TAG=$TAG EXTRA_ARGS="--set train.seed=$SEED" bash scripts/pod_pipeline.sh experiments/07_transformer_crosscon/configs/crosscon_final.yaml
