#!/usr/bin/env bash
# Unattended pod pipeline: wait for the cache build, train (MLflow + CSV logging), evaluate
# (dev: best checkpoint on val + test; final: last checkpoint on test), then stop the pod so
# the GPU stops billing.
# The network volume (/workspace: cache, runs, logs) survives the stop.
#
#   tmux new-session -d -s train "bash scripts/pod_pipeline.sh experiments/03_final_recipe/configs/final_vs.yaml"
#
# Env: AUTO_STOP=0 keeps the pod running afterwards; EXTRA_ARGS passed to train.py
#      (e.g. "--set split=final"); TAG=seed1 adds --tag seed1 (run dir <name>_<split>_seed1).
# The body is one function, so bash parses the whole file before running anything: the
# `git pull` below can then update this script without corrupting the run in progress.
set -uo pipefail

main() {
CONFIG=${1:-experiments/01_reproduction/configs/baseline_vs.yaml}
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

git pull -q || echo "[$(stamp)] git pull failed; using the checked-out commit"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "[$(stamp)] training $CONFIG at commit $(git rev-parse --short HEAD) on $NGPU GPU(s)"
if [ "$NGPU" -gt 1 ]; then  # same recipe: the global batch is split across GPUs (see train.py)
  $PY -m torch.distributed.run --standalone --nproc_per_node="$NGPU" scripts/train.py --config "$CONFIG" ${EXTRA_ARGS:-} ${TAG:+--tag "$TAG"}
else
  $PY -u scripts/train.py --config "$CONFIG" ${EXTRA_ARGS:-} ${TAG:+--tag "$TAG"}
fi
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
)${TAG:+_$TAG}
# dev runs: best.pt (picked on validation) scored on val and, as a read-only diagnostic, test.
# final runs have no validation set, so no best.pt: score last.pt on test.
if [ -f "$RUN_DIR/best.pt" ]; then
  $PY -u scripts/evaluate.py --run "$RUN_DIR" --ckpt best --split val
  $PY -u scripts/evaluate.py --run "$RUN_DIR" --ckpt best --split test
  # per-utterance CTC log-probs for offline LM decoding / candidate combination
  $PY -u scripts/dump_logprobs.py --run "$RUN_DIR" --ckpt best --split val
  $PY -u scripts/dump_logprobs.py --run "$RUN_DIR" --ckpt best --split test
elif [ -f "$RUN_DIR/last.pt" ]; then
  $PY -u scripts/evaluate.py --run "$RUN_DIR" --ckpt last --split test
  $PY -u scripts/dump_logprobs.py --run "$RUN_DIR" --ckpt last --split test
fi

if [ "${AUTO_STOP:-1}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "[$(stamp)] stopping pod $RUNPOD_POD_ID (volume data is kept)"
  sync
  runpodctl stop pod "$RUNPOD_POD_ID" \
    || curl -fsS -X POST -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
    || echo "[$(stamp)] could not stop the pod automatically; stop it from the console"
fi
}

main "$@"
exit $?
