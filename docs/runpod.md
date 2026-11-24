# Running on RunPod

All training and decoding ran on RunPod in EU-RO-1, on one network volume shared by every pod.

## Storage

- **Network volume** `sonispeech` (60 GB, STANDARD, EU-RO-1), mounted at `/workspace`:
  - `cache/v1` (13 GB): the training cache, crop 226:306
  - `cache/audio16k` (1.8 GB): voiced speech at 16 kHz, aligned to the echo frames (step 5b). (`cache/v2_wide`, bins 210:370 for the crop ablations, was deleted on 2026-11-25; rebuild with `preprocess.py --crop 210 370`.)
  - `runs/`: one folder per run, plus `mlruns/` (MLflow)
  - `logs/`, `lm/` (language models + external text), `lisa/` (phase B outputs)
  - `venv/` (training), `venv-decode/` (pyctcdecode + kenlm, numpy < 2), `tools/kenlm/`
  - `sonar-glasses/`: the git checkout every pod runs from
- **Container disk** (wiped on every stop): `/root/venv-llm`, `/root/models/` (Qwen, 15 GB), apt packages.

## Pods used

| Name | GPU | Use |
| --- | --- | --- |
| `sonar-glasses` | RTX PRO 4500 (32 GB), $0.72/h | preprocessing, single-GPU runs, ablations, decoding, LLM |
| `sonar-final-2gpu`, `sonar-seed2` | 2× RTX 4090, $1.78/h | final recipe and seeds (multi-GPU) |

## Running a training job unattended

```bash
cd /workspace/sonar-glasses
tmux new-session -d -s final "bash scripts/pod_pipeline.sh experiments/03_final_recipe/configs/final_vs.yaml \
    2>&1 | tee -a /workspace/logs/train_final.log"
```

`pod_pipeline.sh` pulls the latest code, trains (torchrun if more than one GPU), scores
(dev: best checkpoint on val + test; final: last checkpoint on test), dumps log-probs, then stops
the pod. Env: `TAG=seed1` (run dir suffix), `EXTRA_ARGS="--set train.seed=1"`, `AUTO_STOP=0`.

## Watching from the Mac

```bash
ssh sonar tail -f /workspace/logs/train_final.log      # live log
ssh -t sonar tmux attach -t final                      # the pod terminal (leave with Ctrl-B D, never Ctrl-C)
ssh -N -L 5001:localhost:5000 sonar                    # MLflow at http://localhost:5001
```

MLflow on the pod: `mlflow ui --backend-store-uri file:/workspace/runs/mlruns --port 5000 --workers 4 --gunicorn-opts "--timeout 300"`.
Local port 5001, because macOS uses 5000 for AirPlay.

## Gotchas we hit

- **SSH ports change on every pod restart.** Get the new one from the RunPod console (or the
  proxy: `ssh <pod>@ssh.runpod.io` then `echo $RUNPOD_TCP_PORT_22`) and update `~/.ssh/config`.
- **The checkout is shared by all pods.** A running pipeline re-reads its config and calls
  `scripts/evaluate.py` when training ends, so don't pull a commit that moves those files while
  a run is in progress. Two pods pulling at the same time can also collide on git's lock.
- **Container disk is wiped on stop.** Apt packages vanish, including the Boost libraries KenLM
  needs (`apt-get install libboost-program-options-dev libboost-system-dev libboost-thread-dev`).
- **pyctcdecode pins numpy < 2**, so it lives in `/workspace/venv-decode`, never in the training venv.
- **The image reports host-wide CPUs and RAM.** `preprocess.py` reads the cgroup limits instead.
- **pkill -f matches its own ssh command line.** Kill by PID.
- **eCommons rate-limits** (HTTP 429); curl's retry honours `Retry-After`.
