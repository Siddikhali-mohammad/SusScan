import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class ManualSWA:
    # AveragedModel from torch.optim.swa_utils breaks when the model was compiled,
    # so we do a manual rolling average over the raw state dict instead.
    def __init__(self):
        self.n = 0
        self.avg = None
        self.buffers = None

    @torch.no_grad()
    def update(self, model):
        sd = self._strip(model.state_dict())
        if self.avg is None:
            self.avg     = {k: v.detach().clone().float() for k, v in sd.items() if v.dtype.is_floating_point}
            self.buffers = {k: v.detach().clone()         for k, v in sd.items() if not v.dtype.is_floating_point}
            self.n = 1
        else:
            self.n += 1
            for k in self.avg:
                self.avg[k].add_((sd[k].detach().float() - self.avg[k]) / self.n)

    def state_dict(self):
        out = {k: v.clone() for k, v in self.avg.items()}
        out.update({k: v.clone() for k, v in self.buffers.items()})
        return out

    @staticmethod
    def _strip(sd):
        # torch.compile prefixes everything with "_orig_mod." — strip it
        return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


class SmoothBCEWithLogits(nn.Module):
    # pushes real labels to 0.05 and fake to 0.95 instead of hard 0/1
    def __init__(self, smoothing=0.05, pos_weight=None):
        super().__init__()
        self.s = smoothing
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        t = targets.float()
        t = t * (1 - self.s) + (1 - t) * self.s
        return F.binary_cross_entropy_with_logits(logits, t, pos_weight=self.pos_weight)


def freeze_backbone(params):
    for p in params:
        p.requires_grad = False

def unfreeze_backbone(params):
    for p in params:
        p.requires_grad = True


def build_optimizer_and_scheduler(backbone, head, backbone_lr, head_lr,
                                   weight_decay, total_epochs, steps_per_epoch,
                                   warmup_epochs=1):
    opt = torch.optim.AdamW(
        [
            {"params": backbone, "lr": backbone_lr, "name": "backbone"},
            {"params": head,     "lr": head_lr,     "name": "head"},
        ],
        weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8,
    )

    total_steps  = total_epochs  * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # cosine decay from 1.0 down to 0.01
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    return opt, scheduler


def train_one_epoch(model, loader, opt, scheduler, criterion, device,
                    grad_clip=1.0, amp_dtype=torch.bfloat16, log_every=100):
    model.train()
    running_loss = correct = n = 0

    pbar = tqdm(loader, desc="train", leave=False, ncols=100)
    for step, (imgs, lbls) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with autocast(dtype=amp_dtype):
            logits = model(imgs)
            loss   = criterion(logits, lbls)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        scheduler.step()

        with torch.no_grad():
            correct      += ((logits > 0).float() == lbls).sum().item()
            running_loss += loss.item() * lbls.size(0)
            n            += lbls.size(0)

        if step % log_every == 0:
            pbar.set_postfix(
                loss=f"{running_loss/max(n,1):.4f}",
                acc=f"{correct/max(n,1):.3f}",
                lr=f"{opt.param_groups[0]['lr']:.2e}",
            )

    return {"train_loss": running_loss / max(n, 1), "train_acc": correct / max(n, 1)}


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, amp_dtype=torch.bfloat16):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = n = 0

    for batch in loader:
        imgs = batch[0].to(device, non_blocking=True)
        lbls = batch[1].to(device, non_blocking=True)
        with autocast(dtype=amp_dtype):
            logits = model(imgs)
            loss   = criterion(logits, lbls)
        total_loss += loss.item() * lbls.size(0)
        n          += lbls.size(0)
        all_logits.append(logits.float().cpu())
        all_labels.append(lbls.float().cpu())

    lg    = torch.cat(all_logits).numpy()
    lb    = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-lg))

    auroc = float(roc_auc_score(lb, probs)) if len(np.unique(lb)) > 1 else float("nan")
    return {
        "loss":      total_loss / max(n, 1),
        "acc":       float(accuracy_score(lb, (probs > 0.5).astype(int))),
        "auroc":     auroc,
        "n_samples": n,
    }


@torch.no_grad()
def recompute_bn_stats(model, loader, device, n_batches=200, amp_dtype=torch.bfloat16):
    # SWA averages parameters but BN running stats are stale after averaging.
    # Have to do a fresh forward pass to fix them.
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        with autocast(dtype=amp_dtype):
            model(batch[0].to(device, non_blocking=True))

    model.eval()


def load_trained_model(builder, ckpt_path, device):
    m, _, _ = builder(pretrained=False)
    sd = torch.load(ckpt_path, map_location=device)
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    return m.to(device).eval()


def train_one_model(model_name, model_builder, train_loader, val_loader,
                    total_epochs, swa_window, backbone_lr, head_lr, warmup_epochs,
                    holdout_name, ckpt_dir, log_dir, device,
                    label_smoothing=0.05, weight_decay=0.01, grad_clip_norm=1.0,
                    amp_dtype=torch.bfloat16, use_compile=False, seed=42):

    save_dir = Path(ckpt_dir) / f"holdout_{holdout_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_folder = Path(log_dir) / f"holdout_{holdout_name}"
    log_folder.mkdir(parents=True, exist_ok=True)
    log_path = log_folder / f"{model_name}_epoch_log.jsonl"

    print(f"\n{'='*60}")
    print(f"  {model_name.upper()} | holdout={holdout_name}  epochs={total_epochs}")
    print(f"  lr backbone={backbone_lr}  head={head_lr}  swa_window={swa_window}  compile={use_compile}")

    model, backbone_params, head_params = model_builder()
    model = model.to(device)

    if use_compile:
        print("  compiling... (~60s the first time)")
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)

    opt, sched = build_optimizer_and_scheduler(
        backbone_params, head_params, backbone_lr, head_lr,
        weight_decay=weight_decay,
        total_epochs=total_epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=warmup_epochs,
    )

    n_real = int((train_loader.dataset.labels == 0).sum())
    n_fake = int((train_loader.dataset.labels == 1).sum())
    pos_w  = torch.tensor([n_real / max(n_fake, 1)], device=device)
    print(f"  real={n_real:,}  fake={n_fake:,}  pos_weight={pos_w.item():.3f}")

    criterion  = SmoothBCEWithLogits(label_smoothing, pos_w)
    eval_crit  = nn.BCEWithLogitsLoss()
    swa        = ManualSWA()
    swa_start  = total_epochs - swa_window
    best_auroc = -1.0
    best_epoch = -1
    history    = []

    if warmup_epochs > 0:
        freeze_backbone(backbone_params)

    for epoch in range(total_epochs):
        t0 = time.time()

        if epoch == warmup_epochs and warmup_epochs > 0:
            print(f"  [ep {epoch}] unfreezing backbone")
            unfreeze_backbone(backbone_params)

        train_metrics = train_one_epoch(
            model, train_loader, opt, sched, criterion,
            device=device, grad_clip=grad_clip_norm, amp_dtype=amp_dtype,
        )
        val_metrics = evaluate_model(model, val_loader, eval_crit, device, amp_dtype)

        in_swa = epoch >= swa_start
        if in_swa:
            swa.update(model)

        elapsed = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        log_row = {
            "epoch":          epoch,
            "train_loss":     train_metrics["train_loss"],
            "train_acc":      train_metrics["train_acc"],
            "val_loss":       val_metrics["loss"],
            "val_acc":        val_metrics["acc"],
            "val_auroc":      val_metrics["auroc"],
            "lr_backbone":    opt.param_groups[0]["lr"],
            "lr_head":        opt.param_groups[1]["lr"],
            "epoch_time_sec": elapsed,
            "gpu_mem_peak_gb": peak_gb,
            "in_swa_window":  in_swa,
        }
        history.append(log_row)
        with open(log_path, "a") as f:
            f.write(json.dumps(log_row) + "\n")

        swa_tag = "  [SWA]" if in_swa else ""
        print(f"  [ep {epoch:2d}/{total_epochs-1}]  loss={train_metrics['train_loss']:.4f}"
              f"  val_auc={val_metrics['auroc']:.4f}  acc={val_metrics['acc']:.3f}"
              f"  {elapsed:.0f}s  {peak_gb:.1f}GB{swa_tag}")

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_epoch = epoch
            torch.save(ManualSWA._strip(model.state_dict()), save_dir / f"{model_name}_best.pt")

    # save SWA weights
    torch.save(swa.state_dict(), save_dir / f"{model_name}_swa.pt")

    # pick winner: best checkpoint vs SWA weights
    print("\n  checking best vs swa...")
    fresh, _, _ = model_builder()
    fresh = fresh.to(device)

    candidates = {}
    fresh.load_state_dict(torch.load(save_dir / f"{model_name}_best.pt", map_location=device))
    candidates["best"] = evaluate_model(fresh, val_loader, eval_crit, device, amp_dtype)

    fresh.load_state_dict(swa.state_dict())
    if any(isinstance(m, nn.BatchNorm2d) for m in fresh.modules()):
        print("  fixing BN stats for SWA...")
        recompute_bn_stats(fresh, train_loader, device, amp_dtype=amp_dtype)
    candidates["swa"] = evaluate_model(fresh, val_loader, eval_crit, device, amp_dtype)
    torch.save(fresh.state_dict(), save_dir / f"{model_name}_swa.pt")

    for cname, cm in candidates.items():
        print(f"    {cname}: auc={cm['auroc']:.4f}  acc={cm['acc']:.4f}")

    winner = max(candidates, key=lambda k: candidates[k]["auroc"])
    print(f"  winner -> {winner}")

    del model, fresh, swa, opt, sched
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_name":    model_name,
        "holdout_name":  holdout_name,
        "best_epoch":    best_epoch,
        "history":       history,
        "candidates":    {k: {"auroc": v["auroc"], "acc": v["acc"]} for k, v in candidates.items()},
        "winner":        winner,
        "winner_path":   str(save_dir / f"{model_name}_{winner}.pt"),
        "all_paths":     {
            "best": str(save_dir / f"{model_name}_best.pt"),
            "swa":  str(save_dir / f"{model_name}_swa.pt"),
        },
        "log_path":      str(log_path),
    }
