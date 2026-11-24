#!/usr/bin/env python3
"""Train the ResNet-34 + CTC baseline (README section 8).

  python scripts/train.py --config experiments/01_reproduction/configs/baseline_vs.yaml
  python scripts/train.py --config experiments/01_reproduction/configs/baseline_vs.yaml --set split=final
  python scripts/train.py --config experiments/01_reproduction/configs/baseline_vs.yaml --smoke 10 --set train.epochs=5
  python scripts/train.py --config experiments/01_reproduction/configs/baseline_vs.yaml --overfit 20 --set train.epochs=150

Run directory <runs_dir>/<run_name>_<split>[_tag]/ holds config.yaml, git commit, tokenizer,
normalisation stats, metrics.csv, last.pt (every epoch, used to resume) and best.pt
(lowest mean validation WER). Rerunning the same command resumes from last.pt.
Logged per step: train/loss, train/lr. Per epoch: epoch, train/loss_epoch, train/frames_per_s,
time/epoch_s, gpu/mem_gb; at validation: val/loss and val/wer_{voiced,silent,mean}.

Multi-GPU (same recipe, faster): torchrun --standalone --nproc_per_node=N scripts/train.py ...
Each global batch of train.batch_size is split across the N GPUs and gradients are averaged,
so the update equals single-GPU training with the same batch (the model uses GroupNorm, so
there are no batch statistics to synchronise). Only rank 0 logs, evaluates and saves.
"""
import os
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
import torch.distributed as dist
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import (BucketBatchSampler, EchoDataset, collate, compute_stats, load_rows,  # noqa: E402
                  preload, save_json)
from decode import Tokenizer, greedy_decode, train_tokenizer, wer  # noqa: E402
from model import CrossConHead, ResNet34CTC, build_model  # noqa: E402
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
    if os.environ.get("SONAR_DEVICE"):  # e.g. cpu, to test the multi-process path on a laptop
        return torch.device(os.environ["SONAR_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_lambda(total_steps, warmup_frac, final_frac, kind="exp"):
    warm = max(1, int(total_steps * warmup_frac))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, total_steps - warm))
        if kind == "cosine":  # cosine from 1 to final_frac
            return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * progress))
        return final_frac ** progress  # exponential decay from 1 to final_frac
    return f


def ctc_loss_fn(log_probs, out_lens, targets, target_lens, ctc, device):
    lp = log_probs.transpose(0, 1)  # (T', B, C)
    if device.type == "mps":        # CTC is not implemented on MPS
        lp, out_lens, targets, target_lens = lp.cpu(), out_lens.cpu(), targets.cpu(), target_lens.cpu()
    return ctc(lp, targets, out_lens, target_lens)


def ctc_sum_per_sample(log_probs, out_lens, targets, target_lens, device):
    """Sum over the batch of each sample's CTC loss divided by its target length: the numerator
    of nn.CTCLoss(reduction="mean"), so sum / global batch size reproduces the single-GPU loss."""
    ctc = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")
    per = ctc_loss_fn(log_probs, out_lens, targets, target_lens, ctc, device)
    return (per / target_lens.to(per.device).clamp(min=1)).sum()


class ShardedBatches(torch.utils.data.Sampler):
    """Rank r of N takes batch[r::N] of every global batch. Batches smaller than N (at most the
    one partial batch per pool) are skipped so every rank runs the same number of steps."""

    def __init__(self, batch_sampler, rank, world):
        self.bs, self.rank, self.world = batch_sampler, rank, world

    def set_epoch(self, epoch):
        self.bs.set_epoch(epoch)

    def __iter__(self):
        for b in self.bs:
            if len(b) >= self.world:
                yield b[self.rank::self.world]

    def __len__(self):
        return sum(1 for b in self.bs if len(b) >= self.world)


@torch.no_grad()
def evaluate(model, loader, tok, ctc, device, amp_dtype):
    model.eval()
    losses, n = 0.0, 0
    by_mode = {}
    for x, lens, y, ylens, texts, modes in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            lp = model(x, lens)
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
    cache = Path(cfg["cache"])

    # ---- distributed setup (torchrun sets WORLD_SIZE / RANK / LOCAL_RANK)
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    ddp = world > 1
    device = pick_device()
    if ddp:
        if device.type == "cuda":
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    main_proc = rank == 0

    def barrier():
        if ddp:
            dist.barrier()

    def log(msg):
        if main_proc:
            print(msg, flush=True)

    if main_proc:
        run_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    random.seed(tc["seed"])
    np.random.seed(tc["seed"])
    torch.manual_seed(tc["seed"])
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(str(tc.get("amp")).lower()) if device.type == "cuda" else None
    log(f"run {run_name} on {device} x{world} (amp {amp_dtype}), commit {git_commit()[:7]}")

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
    if main_proc:
        if not tok_path.exists():
            all_train = load_rows(cache, cfg["split"], "train", ["voiced", "silent"])
            texts = sorted({r["text"] for r in all_train})
            print(f"training tokenizer on {len(texts)} unique sentences", flush=True)
            train_tokenizer(texts, tok_path.with_suffix(""), cfg["tokenizer"]["vocab_size"])
        shutil.copy(tok_path, run_dir / "tokenizer.model")
    barrier()
    tok = Tokenizer(run_dir / "tokenizer.model")

    t0 = time.time()
    train_arrays = preload(cache, train_rows, cfg.get("crop"))
    val_arrays = preload(cache, val_rows, cfg.get("crop")) if val_rows else []
    log(f"loaded {len(train_rows)} train / {len(val_rows)} val utterances in {time.time() - t0:.0f} s")

    stats_path = run_dir / "stats.json"
    if main_proc and not stats_path.exists():
        if cfg.get("normalize", True):
            stats = compute_stats(train_arrays, tc.get("stats_max_utts", 4000), tc["seed"])
        else:  # official code: no normalisation of the profiles
            stats = dict(mean=[0.0] * 4, std=[1.0] * 4, n_values_per_channel=0, n_utts=0)
        save_json(stats, stats_path)
    barrier()
    stats = yaml.safe_load(open(stats_path))
    log(f"normalisation mean {np.round(stats['mean'], 5)} std {np.round(stats['std'], 5)}")

    # CTC needs T_out >= target length for every sample
    worst = min((math.ceil(r["n_frames"] / 16) - len(tok.encode(r["text"])) for r in train_rows))
    assert worst >= 0, f"an utterance is too short for its transcript under CTC (margin {worst})"

    cc = cfg.get("crosscon")  # step 5b: speech-feature targets per output step (make_targets.py)
    targets = None
    if cc:
        def load_target(r):
            f = Path(cc["dir"]) / r["path"]
            return np.load(f) if f.exists() else None
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(16) as ex:
            targets = list(ex.map(load_target, train_rows))
        n_t = sum(t is not None for t in targets)
        log(f"crossCon: targets for {n_t}/{len(train_rows)} training utterances "
            f"(voiced {sum(t is not None for t, r in zip(targets, train_rows) if r['mode'] == 'voiced')}, "
            f"silent {sum(t is not None for t, r in zip(targets, train_rows) if r['mode'] == 'silent')}), "
            f"weight {cc.get('weight', 1.0)}, temperature {cc.get('temperature', 0.1)}")
    train_ds = EchoDataset(train_rows, train_arrays, tok, stats, train=not args.overfit, aug=aug, targets=targets)
    sampler = BucketBatchSampler([r["n_frames"] for r in train_rows], tc["batch_size"], seed=tc["seed"])
    if ddp:
        sampler = ShardedBatches(sampler, rank, world)
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
    raw_model = build_model(cfg["model"], tok.n_classes).to(device)
    model = raw_model
    raw_head = head = None
    if cc:
        raw_head = CrossConHead(raw_model.fc.in_features, train_ds.target_dim, cc.get("proj_dim", 256),
                                cc.get("temperature", 0.1)).to(device)
        head = raw_head
    params = list(raw_model.parameters()) + (list(raw_head.parameters()) if raw_head else [])
    opt_cls = torch.optim.AdamW if tc.get("optimizer", "adam") == "adamw" else torch.optim.Adam
    opt = opt_cls(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    total_steps = tc["epochs"] * len(sampler)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(total_steps, tc["warmup_frac"], tc["final_lr_frac"], tc.get("schedule", "exp")))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    n_params = sum(p.numel() for p in model.parameters())

    start_epoch, step, best, run_id = 0, 0, float("inf"), None
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        raw_model.load_state_dict(ck["model"])
        if raw_head is not None and "cc_head" in ck:
            raw_head.load_state_dict(ck["cc_head"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step, best, run_id = ck["epoch"] + 1, ck["step"], ck["best"], ck.get("run_id")
        log(f"resumed from epoch {ck['epoch']} (step {step}, best mean WER {best:.4f})")
    if ddp:  # wraps after loading; DDP broadcasts rank 0's weights to all ranks
        model = torch.nn.parallel.DistributedDataParallel(
            raw_model, device_ids=[device.index] if device.type == "cuda" else None)
        if raw_head is not None:
            head = torch.nn.parallel.DistributedDataParallel(
                raw_head, device_ids=[device.index] if device.type == "cuda" else None)
        # different augmentation randomness per rank (DataLoader workers seed from torch's RNG)
        random.seed(tc["seed"] + rank)
        torch.manual_seed(tc["seed"] + rank)

    meta = dict(cfg=cfg, commit=git_commit(), run_name=run_name, n_params=n_params, device=str(device),
                n_gpus=world, n_train=len(train_rows), n_val=len(val_rows), total_steps=total_steps,
                smoke=args.smoke, overfit=args.overfit)
    tracker = None
    if main_proc:
        yaml.safe_dump(meta, open(run_dir / "config.yaml", "w"), sort_keys=False)
        tracker = Tracker(cfg["log"]["backend"], run_dir, run_name, meta, cfg["log"].get("project", "sonar-glasses"),
                          resume_id=run_id)
    log(f"{n_params / 1e6:.1f} M params, {len(sampler)} steps/epoch, {total_steps} total steps")

    def all_sum(*vals):
        """Sum scalars over ranks (identity on one process)."""
        t = torch.tensor(vals, dtype=torch.float64, device=device)
        if ddp:
            dist.all_reduce(t)
        return t.tolist()

    # ---- train
    model.train()
    for epoch in range(start_epoch, tc["epochs"]):
        sampler.set_epoch(epoch)
        t_ep, loss_sum, n_batches, frames = time.time(), 0.0, 0, 0
        nce_sum = 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for batch in train_dl:
            x, lens, y, ylens = batch[:4]
            x = x.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                if head is not None:
                    lp, hid = model(x, lens, return_hidden=True)
                else:
                    lp = model(x, lens)
            out_lens = ResNet34CTC.out_lengths(lens).to(device)
            local_sum = ctc_sum_per_sample(lp, out_lens, y.to(device), ylens.to(device), device)
            n_global = all_sum(len(lens))[0]
            # DDP averages gradients over ranks, so scale by world / global batch size: the update
            # then equals the single-GPU mean loss over the whole global batch
            loss = local_sum * (world / n_global)
            if head is not None:  # crossCon InfoNCE (mean over this rank's valid steps)
                nce, _ = head(hid, batch[6].to(device, non_blocking=True), batch[7].to(device, non_blocking=True),
                              cc.get("max_pairs", 2048))
                nce_sum += float(nce.detach())
                loss = loss + cc.get("weight", 1.0) * nce
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            batch_loss = (float(loss.detach()) - (cc.get("weight", 1.0) * float(nce.detach()) if head is not None else 0.0)) / world  # CTC part: this rank's share of the global mean loss
            loss_sum += batch_loss
            n_batches += 1
            frames += int(lens.sum())
            if step % 25 == 0:
                step_loss = all_sum(batch_loss)[0]
                if main_proc:
                    tracker.log({"train/loss": step_loss, "train/lr": sched.get_last_lr()[0],
                                 "train/grad_norm": float(gnorm), "epoch": epoch}, step)
        ep_time = time.time() - t_ep
        loss_sum, frames = all_sum(loss_sum, frames)
        metrics = {"epoch": epoch, "train/loss_epoch": loss_sum / max(n_batches, 1), "train/lr": sched.get_last_lr()[0],
                   "time/epoch_s": ep_time, "train/frames_per_s": frames / ep_time}
        if head is not None:
            metrics["train/crosscon_epoch"] = all_sum(nce_sum)[0] / world / max(n_batches, 1)
        if device.type == "cuda":
            metrics["gpu/mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        is_eval = val_dl is not None and ((epoch + 1) % cfg["eval"]["every"] == 0 or epoch + 1 == tc["epochs"]
                                          or args.overfit or args.smoke)
        if not main_proc:  # rank 0 evaluates and saves; the others wait for it
            barrier()
            if tc.get("stop_after") and epoch + 1 >= int(tc["stop_after"]):
                break
            continue
        if is_eval:
            vm, examples = evaluate(raw_model, val_dl, tok, ctc, device, amp_dtype)
            metrics.update(vm)
            for m, ex in examples.items():
                for ref, hyp in ex[:2]:
                    print(f"   [{m}] REF: {ref}\n   [{m}] HYP: {hyp}", flush=True)
        tracker.log(metrics, step)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()), flush=True)

        ck = dict(model=raw_model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                  epoch=epoch, step=step, best=best, run_id=tracker.run_id, cfg=cfg, n_classes=tok.n_classes)
        if raw_head is not None:
            ck["cc_head"] = raw_head.state_dict()
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
        barrier()
        if tc.get("stop_after") and epoch + 1 >= int(tc["stop_after"]):
            print(f"stop_after={tc['stop_after']} reached; stopping (schedule length {tc['epochs']} epochs)", flush=True)
            break

    if main_proc:
        tracker.finish()
        print(f"done. best mean val WER {best:.4f}; run dir {run_dir}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
