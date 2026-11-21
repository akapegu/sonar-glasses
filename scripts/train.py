#!/usr/bin/env python3
"""Train the ResNet-34 + CTC baseline (README section 8).

  python scripts/train.py --config configs/baseline_vs.yaml
  python scripts/train.py --config configs/baseline_vs.yaml --set split=final
  python scripts/train.py --config configs/baseline_vs.yaml --smoke 10 --set train.epochs=5
  python scripts/train.py --config configs/baseline_vs.yaml --overfit 20 --set train.epochs=150

Run directory <runs_dir>/<run_name>_<split>[_tag]/ holds config.yaml, git commit, tokenizer,
normalisation stats, metrics.csv, last.pt (every epoch, used to resume) and best.pt
(lowest mean validation WER). Rerunning the same command resumes from last.pt.
Logged per step: train/loss, train/lr. Per epoch: epoch, train/loss_epoch, train/frames_per_s,
time/epoch_s, gpu/mem_gb; at validation: val/loss and val/wer_{voiced,silent,mean}.
"""
import argparse
import copy
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import (BucketBatchSampler, EchoDataset, collate, compute_stats, load_rows,  # noqa: E402
                  preload, save_json)
from decode import Tokenizer, greedy_decode, train_tokenizer, wer  # noqa: E402
from model import ResNet34CTC  # noqa: E402
from tracking import Tracker  # noqa: E402


def set_dotted(cfg, key, value):
    d = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    v = yaml.safe_load(value)
    if isinstance(v, str):  # YAML 1.1 reads "1e-3" as a string; accept it as a number
        try:
            v = float(v)
        except ValueError:
            pass
    d[parts[-1]] = v


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_lambda(total_steps, warmup_frac, final_frac):
    warm = max(1, int(total_steps * warmup_frac))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, total_steps - warm))
        return final_frac ** progress  # exponential decay from 1 to final_frac
    return f


def ctc_loss_fn(log_probs, out_lens, targets, target_lens, ctc, device):
    lp = log_probs.transpose(0, 1)  # (T', B, C)
    if device.type == "mps":        # CTC is not implemented on MPS
        lp, out_lens, targets, target_lens = lp.cpu(), out_lens.cpu(), targets.cpu(), target_lens.cpu()
    return ctc(lp, targets, out_lens, target_lens)


@torch.no_grad()
def evaluate(model, loader, tok, ctc, device, amp_dtype):
    model.eval()
    losses, n = 0.0, 0
    by_mode = {}
    for x, lens, y, ylens, texts, modes in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            lp = model(x)
        out_lens = ResNet34CTC.out_lengths(lens).to(device)
        loss = ctc_loss_fn(lp, out_lens, y.to(device), ylens.to(device), ctc, device)
        losses += float(loss) * len(texts)
        n += len(texts)
        for t, h, m in zip(texts, greedy_decode(lp, out_lens, tok), modes):
            by_mode.setdefault(m, ([], []))
            by_mode[m][0].append(t)
            by_mode[m][1].append(h)
    model.train()
    out = {"val/loss": losses / max(n, 1)}
    for m, (refs, hyps) in by_mode.items():
        out[f"val/wer_{m}"] = wer(refs, hyps)
    wers = [v for k, v in out.items() if k.startswith("val/wer_")]
    out["val/wer_mean"] = float(np.mean(wers)) if wers else float("nan")
    examples = {m: list(zip(r[:3], h[:3])) for m, (r, h) in by_mode.items()}
    return out, examples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.epochs=5 cache=cache/v1")
    ap.add_argument("--smoke", type=int, help="train on the first N train sessions (both modes) only")
    ap.add_argument("--overfit", type=int, help="overfit the first N train utterances, no augmentation")
    ap.add_argument("--tag", default="", help="suffix for the run directory")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for kv in args.set:
        k, v = kv.split("=", 1)
        set_dotted(cfg, k, v)
    tc, aug = cfg["train"], copy.deepcopy(cfg["aug"])
    tag = args.tag or ("smoke" if args.smoke else (f"overfit{args.overfit}" if args.overfit else ""))
    run_name = f"{cfg['run_name']}_{cfg['split']}" + (f"_{tag}" if tag else "")
    run_dir = Path(cfg["runs_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg["cache"])

    random.seed(tc["seed"])
    np.random.seed(tc["seed"])
    torch.manual_seed(tc["seed"])
    device = pick_device()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(str(tc.get("amp")).lower()) if device.type == "cuda" else None
    print(f"run {run_name} on {device} (amp {amp_dtype}), commit {git_commit()[:7]}", flush=True)

    # ---- data
    train_rows = load_rows(cache, cfg["split"], "train", cfg["modes"])
    if args.smoke:
        keep = sorted({r["session"] for r in train_rows})[: args.smoke]
        train_rows = [r for r in train_rows if r["session"] in keep]
    if args.overfit:
        train_rows = train_rows[: args.overfit]
        aug = {k: 0 for k in aug}
    val_rows = load_rows(cache, cfg["split"], "val", ["voiced", "silent"]) if cfg["split"] == "dev" else []
    if args.overfit:
        val_rows = train_rows
    if not train_rows:
        sys.exit("no training rows: check cache/manifest.csv and the split")

    # leakage guard: no validation sentence may appear in training (README section 4)
    overlap = {r["text"] for r in train_rows} & {r["text"] for r in val_rows}
    if overlap and not args.overfit:
        sys.exit(f"{len(overlap)} validation sentences also appear in training")

    # ---- tokenizer (trained once per split on that split's training transcripts)
    tok_path = cache / "tokenizers" / f"{cfg['split']}_unigram{cfg['tokenizer']['vocab_size']}.model"
    if not tok_path.exists():
        all_train = load_rows(cache, cfg["split"], "train", ["voiced", "silent"])
        texts = sorted({r["text"] for r in all_train})
        print(f"training tokenizer on {len(texts)} unique sentences", flush=True)
        train_tokenizer(texts, tok_path.with_suffix(""), cfg["tokenizer"]["vocab_size"])
    shutil.copy(tok_path, run_dir / "tokenizer.model")
    tok = Tokenizer(run_dir / "tokenizer.model")

    t0 = time.time()
    train_arrays = preload(cache, train_rows, cfg.get("crop"))
    val_arrays = preload(cache, val_rows, cfg.get("crop")) if val_rows else []
    print(f"loaded {len(train_rows)} train / {len(val_rows)} val utterances in {time.time() - t0:.0f} s", flush=True)

    stats_path = run_dir / "stats.json"
    if stats_path.exists():
        stats = yaml.safe_load(open(stats_path))
    else:
        if cfg.get("normalize", True):
            stats = compute_stats(train_arrays, tc.get("stats_max_utts", 4000), tc["seed"])
        else:  # official code: no normalisation of the profiles
            stats = dict(mean=[0.0] * 4, std=[1.0] * 4, n_values_per_channel=0, n_utts=0)
        save_json(stats, stats_path)
    print(f"normalisation mean {np.round(stats['mean'], 5)} std {np.round(stats['std'], 5)}", flush=True)

    # CTC needs T_out >= target length for every sample
    worst = min((math.ceil(r["n_frames"] / 16) - len(tok.encode(r["text"])) for r in train_rows))
    assert worst >= 0, f"an utterance is too short for its transcript under CTC (margin {worst})"

    train_ds = EchoDataset(train_rows, train_arrays, tok, stats, train=not args.overfit, aug=aug)
    sampler = BucketBatchSampler([r["n_frames"] for r in train_rows], tc["batch_size"], seed=tc["seed"])
    nw = tc.get("num_workers", 4) if device.type == "cuda" else 0
    train_dl = torch.utils.data.DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=nw,
                                           pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    val_dl = None
    if val_rows:
        val_ds = EchoDataset(val_rows, val_arrays, tok, stats, train=False, aug={})
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_sampler=BucketBatchSampler([r["n_frames"] for r in val_rows], tc["batch_size"], shuffle=False),
            collate_fn=collate, num_workers=min(nw, 4), pin_memory=device.type == "cuda")

    # ---- model / optimiser / schedule
    model = ResNet34CTC(n_classes=tok.n_classes, range_stride=cfg["model"]["range_stride"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total_steps = tc["epochs"] * len(sampler)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total_steps, tc["warmup_frac"], tc["final_lr_frac"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    n_params = sum(p.numel() for p in model.parameters())

    start_epoch, step, best, run_id = 0, 0, float("inf"), None
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step, best, run_id = ck["epoch"] + 1, ck["step"], ck["best"], ck.get("run_id")
        print(f"resumed from epoch {ck['epoch']} (step {step}, best mean WER {best:.4f})", flush=True)

    meta = dict(cfg=cfg, commit=git_commit(), run_name=run_name, n_params=n_params, device=str(device),
                n_train=len(train_rows), n_val=len(val_rows), total_steps=total_steps, smoke=args.smoke,
                overfit=args.overfit)
    yaml.safe_dump(meta, open(run_dir / "config.yaml", "w"), sort_keys=False)
    tracker = Tracker(cfg["log"]["backend"], run_dir, run_name, meta, cfg["log"].get("project", "sonar-glasses"),
                      resume_id=run_id)
    print(f"{n_params / 1e6:.1f} M params, {len(sampler)} steps/epoch, {total_steps} total steps", flush=True)

    # ---- train
    model.train()
    for epoch in range(start_epoch, tc["epochs"]):
        sampler.set_epoch(epoch)
        t_ep, loss_sum, n_batches, frames = time.time(), 0.0, 0, 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for x, lens, y, ylens, _, _ in train_dl:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                lp = model(x)
            out_lens = ResNet34CTC.out_lengths(lens).to(device)
            loss = ctc_loss_fn(lp, out_lens, y.to(device), ylens.to(device), ctc, device)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            loss_sum += float(loss.detach())
            n_batches += 1
            frames += int(lens.sum())
            if step % 25 == 0:
                tracker.log({"train/loss": float(loss.detach()), "train/lr": sched.get_last_lr()[0],
                             "train/grad_norm": float(gnorm), "epoch": epoch}, step)
        ep_time = time.time() - t_ep
        metrics = {"epoch": epoch, "train/loss_epoch": loss_sum / max(n_batches, 1), "train/lr": sched.get_last_lr()[0],
                   "time/epoch_s": ep_time, "train/frames_per_s": frames / ep_time}
        if device.type == "cuda":
            metrics["gpu/mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        is_eval = val_dl is not None and ((epoch + 1) % cfg["eval"]["every"] == 0 or epoch + 1 == tc["epochs"]
                                          or args.overfit or args.smoke)
        if is_eval:
            vm, examples = evaluate(model, val_dl, tok, ctc, device, amp_dtype)
            metrics.update(vm)
            for m, ex in examples.items():
                for ref, hyp in ex[:2]:
                    print(f"   [{m}] REF: {ref}\n   [{m}] HYP: {hyp}", flush=True)
        tracker.log(metrics, step)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()), flush=True)

        ck = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                  epoch=epoch, step=step, best=best, run_id=tracker.run_id, cfg=cfg, n_classes=tok.n_classes)
        if is_eval and metrics["val/wer_mean"] < best:
            best = metrics["val/wer_mean"]
            ck["best"] = best
            torch.save(ck, run_dir / "best.pt")
            print(f"   new best mean val WER {best:.4f} (epoch {epoch})", flush=True)
        tmp = run_dir / "last.pt.tmp"
        torch.save(ck, tmp)
        tmp.replace(last)
        # snapshot fixed epochs (model weights only) so short ablations and full runs can be
        # compared at the same epoch on any split, e.g. epoch39.pt -> evaluate.py --ckpt epoch39
        if epoch in tc.get("keep_epochs", [39]):
            torch.save({k: v for k, v in ck.items() if k not in ("opt", "sched", "scaler")},
                       run_dir / f"epoch{epoch}.pt")
        # short ablations: keep the full-length LR schedule (train.epochs) but stop early, so the
        # curve is directly comparable epoch-for-epoch with a full run
        if tc.get("stop_after") and epoch + 1 >= int(tc["stop_after"]):
            print(f"stop_after={tc['stop_after']} reached; stopping (schedule length {tc['epochs']} epochs)", flush=True)
            break

    tracker.finish()
    print(f"done. best mean val WER {best:.4f}; run dir {run_dir}", flush=True)


if __name__ == "__main__":
    main()
