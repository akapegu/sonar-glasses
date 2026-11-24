#!/usr/bin/env bash
# Run a queue of short dev-split ablations one after another, then stop the pod.
# Each run keeps the full 200-epoch LR schedule but stops after STOP_AFTER epochs, so its
# validation curve is comparable epoch-for-epoch with baseline_vs_dev.
#
#   tmux new-session -d -s ablate "bash experiments/02_ablations/ablate.sh experiments/01_reproduction/configs/baseline_vs.yaml \
#       'rs1222|model.range_stride=[1,2,2,2]' 'aug2|aug.time_masks=6 aug.range_mask_max=16' \
#       2>&1 | tee -a /workspace/logs/ablate.log"
#
# Each queue item is "tag|override override ...". Env: STOP_AFTER (default 40), AUTO_STOP=0
# keeps the pod running afterwards. A failed run is logged and the queue moves on.
set -uo pipefail

main() {
CONFIG=$1; shift
cd "$(dirname "$0")/../.."
PY=/workspace/venv/bin/python
STOP_AFTER=${STOP_AFTER:-40}
[ -f /etc/rp_environment ] && source /etc/rp_environment
stamp() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }

git pull -q || echo "[$(stamp)] git pull failed; using the checked-out commit"
echo "[$(stamp)] ablation queue at commit $(git rev-parse --short HEAD): $*"
export PYTHONUNBUFFERED=1
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NGPU" -gt 1 ]; then TRAIN="$PY -m torch.distributed.run --standalone --nproc_per_node=$NGPU"; else TRAIN=$PY; fi
echo "[$(stamp)] training on $NGPU GPU(s)"
TAGS=()
for item in "$@"; do
  tag=${item%%|*}; overrides=${item#*|}
  TAGS+=("$tag")
  echo "[$(stamp)] === $tag: $overrides (stop after $STOP_AFTER)"
  # shellcheck disable=SC2086
  if $TRAIN scripts/train.py --config "$CONFIG" --tag "$tag" --set train.stop_after="$STOP_AFTER" $overrides; then
    run=$($PY -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print(f\"{c['runs_dir']}/{c['run_name']}_dev_$tag\")" "$CONFIG")
    # test scores are for information only: settings are chosen on validation
    for split in val test; do
      $PY -u scripts/evaluate.py --run "$run" --ckpt best --split $split | grep -E '"wer_|"epoch"' | tr -d '\n'; echo "  <- $tag $split"
    done
  else
    echo "[$(stamp)] $tag failed"
  fi
done

echo "[$(stamp)] summary (last validation of each run; baseline = baseline_vs_dev at the same epoch)"
$PY - "$CONFIG" "$STOP_AFTER" "${TAGS[@]}" <<'EOF'
import csv, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])); stop = int(sys.argv[2])
base = f"{cfg['runs_dir']}/{cfg['run_name']}_dev"
def last_val(run, max_epoch=None):
    rows = list(csv.reader(open(f"{run}/metrics.csv")))[1:]
    by_step, epoch_of = {}, {}
    for step, k, v in rows:
        by_step.setdefault(int(step), {})[k] = float(v)
    evals = [(m["epoch"], m) for m in by_step.values() if "val/wer_mean" in m and "epoch" in m
             and (max_epoch is None or m["epoch"] <= max_epoch)]
    return max(evals, key=lambda e: e[0]) if evals else (None, {})
for name, run, cap in [("baseline", base, stop - 1)] + [(t, f"{base}_{t}", None) for t in sys.argv[3:]]:
    try:
        ep, m = last_val(run, cap)
        print(f"{name:12s} epoch {int(ep):3d}  voiced {m['val/wer_voiced']:.1%}  silent {m['val/wer_silent']:.1%}  "
              f"mean {m['val/wer_mean']:.1%}  val_loss {m.get('val/loss', float('nan')):.3f}")
    except Exception as e:  # noqa: BLE001
        print(f"{name:12s} no result ({e})")
EOF

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
