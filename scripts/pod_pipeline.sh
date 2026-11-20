#!/usr/bin/env bash
# Unattended pod pipeline: wait for the cache build, wait (bounded) for the W&B key,
# train, evaluate the best checkpoint on validation, then stop the pod so the GPU stops billing.
# The network volume (/workspace: cache, runs, logs) survives the stop.
#
#   tmux new-session -d -s train "bash scripts/pod_pipeline.sh configs/baseline_vs.yaml"
#
# Env: WAIT_KEY_MIN (default 60) minutes to wait for /workspace/.secrets/wandb_key before
#      falling back to CSV logging; AUTO_STOP=0 keeps the pod running afterwards;
#      EXTRA_ARGS passed to train.py (e.g. "--set split=final").
set -uo pipefail
CONFIG=${1:-configs/baseline_vs.yaml}
cd "$(dirname "$0")/.."
LOG=/workspace/logs
PY=/workspace/venv/bin/python
mkdir -p "$LOG"
[ -f /etc/rp_environment ] && source /etc/rp_environment
stamp() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }

echo "[$(stamp)] waiting for the cache build to finish"
until grep -q "all requested zips done" "$LOG/fetch.log" 2>/dev/null; do
  if grep -qE "Traceback|mismatch" "$LOG/fetch.log" 2>/dev/null; then
    echo "[$(stamp)] cache build failed (see $LOG/fetch.log); not training"; exit 1
  fi
  sleep 60
done
echo "[$(stamp)] cache complete"

for _ in $(seq 1 "${WAIT_KEY_MIN:-60}"); do
  [ -f /workspace/.secrets/wandb_key ] && break
  sleep 60
done
[ -f /workspace/.secrets/wandb_key ] && echo "[$(stamp)] W&B key found" || echo "[$(stamp)] no W&B key; CSV logging only"

git pull -q || echo "[$(stamp)] git pull failed; using the checked-out commit"
echo "[$(stamp)] training $CONFIG at commit $(git rev-parse --short HEAD)"
$PY -u scripts/train.py --config "$CONFIG" ${EXTRA_ARGS:-}
TRAIN_STATUS=$?
echo "[$(stamp)] train.py exited with $TRAIN_STATUS"

RUN_DIR=$($PY - "$CONFIG" ${EXTRA_ARGS:-} <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
args = sys.argv[2:]
for i, a in enumerate(args):
    if a.startswith("split="):
        cfg["split"] = a.split("=", 1)[1]
print(f"{cfg['runs_dir']}/{cfg['run_name']}_{cfg['split']}")
EOF
)
if [ -f "$RUN_DIR/best.pt" ]; then
  $PY -u scripts/evaluate.py --run "$RUN_DIR" --ckpt best --split val
fi

if [ "${AUTO_STOP:-1}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "[$(stamp)] stopping pod $RUNPOD_POD_ID (volume data is kept)"
  sync
  runpodctl stop pod "$RUNPOD_POD_ID" \
    || curl -fsS -X POST -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
    || echo "[$(stamp)] could not stop the pod automatically; stop it from the console"
fi
