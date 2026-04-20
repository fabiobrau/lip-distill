#!/usr/bin/env python3
"""
Gradient-alignment distillation: RobustBench teacher → CPL student.

Training modes (--mode):
  ce_only        Baseline: student trained with CE loss only.
  logit_kd       Baseline: CE + logit knowledge distillation (Hinton).
  clean_align    Ablation: CE + gradient alignment at CLEAN points.
  adv_align      Main method: CE + gradient alignment at ADVERSARIAL points.
  traj_align     Extension: CE + trajectory gradient alignment (all PGD steps).

Example (main method):
  python train.py --mode adv_align --epochs 100 --batch_size 64

Example (fast sanity check):
  python train.py --mode ce_only --epochs 5 --n_channels 64 --n_blocks 2
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data import get_cifar10_loaders
from models import CPLStudent
from attacks import pgd_linf
from losses import (
    teacher_gradient,
    student_gradient,
    gradient_alignment_loss,
    logit_kd_loss,
)


# ---------------------------------------------------------------------------
# Teacher loading
# ---------------------------------------------------------------------------

def load_teacher(model_name: str, device: torch.device) -> nn.Module:
    from robustbench.utils import load_model
    teacher = load_model(model_name=model_name, dataset="cifar10", threat_model="Linf")
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_clean(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def evaluate_pgd(
    model: nn.Module,
    loader,
    device: torch.device,
    eps: float,
    alpha: float,
    n_steps: int = 20,
    n_batches: int = 10,
) -> float:
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        x_adv = pgd_linf(model, x, y, eps=eps, alpha=alpha, n_steps=n_steps)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Training step implementations per mode
# ---------------------------------------------------------------------------

def step_ce_only(student, x, y, **kw):
    loss = F.cross_entropy(student(x), y)
    return loss, {}


def step_logit_kd(student, x, y, teacher, kd_temperature, **kw):
    logits_s = student(x)
    with torch.no_grad():
        logits_t = teacher(x)
    loss_ce = F.cross_entropy(logits_s, y)
    loss_kd = logit_kd_loss(logits_s, logits_t.detach(), temperature=kd_temperature)
    loss = loss_ce + kw["lambda_align"] * loss_kd
    return loss, {"loss_ce": loss_ce.item(), "loss_kd": loss_kd.item()}


def step_clean_align(student, x, y, teacher, lambda_align, **kw):
    # CE on clean data
    loss_ce = F.cross_entropy(student(x), y)

    # Gradient alignment at CLEAN points
    x_clean = x.detach().requires_grad_(True)
    g_t = teacher_gradient(teacher, x_clean, y)
    g_s, _ = student_gradient(student, x_clean, y)
    loss_align = gradient_alignment_loss(g_s, g_t)

    loss = loss_ce + lambda_align * loss_align
    return loss, {"loss_ce": loss_ce.item(), "loss_align": loss_align.item()}


def step_adv_align(student, x, y, teacher, lambda_align, eps, alpha, pgd_steps, **kw):
    # PGD against teacher to find adversarial points
    x_adv = pgd_linf(teacher, x, y, eps=eps, alpha=alpha, n_steps=pgd_steps)

    # CE on clean data
    loss_ce = F.cross_entropy(student(x), y)

    # Gradient alignment at ADVERSARIAL points
    g_t = teacher_gradient(teacher, x_adv, y)
    g_s, _ = student_gradient(student, x_adv, y)
    loss_align = gradient_alignment_loss(g_s, g_t)

    loss = loss_ce + lambda_align * loss_align
    return loss, {"loss_ce": loss_ce.item(), "loss_align": loss_align.item()}


def step_traj_align(student, x, y, teacher, lambda_align, eps, alpha, pgd_steps, **kw):
    # PGD with full trajectory recording
    _, trajectory = pgd_linf(
        teacher, x, y, eps=eps, alpha=alpha, n_steps=pgd_steps,
        return_trajectory=True,
    )

    # CE on clean data
    loss_ce = F.cross_entropy(student(x), y)

    # Alignment at each trajectory point
    align_losses = []
    for x_k in trajectory:
        g_t_k = teacher_gradient(teacher, x_k, y)
        g_s_k, _ = student_gradient(student, x_k, y)
        align_losses.append(gradient_alignment_loss(g_s_k, g_t_k))

    loss_align = torch.stack(align_losses).mean()
    loss = loss_ce + lambda_align * loss_align
    return loss, {"loss_ce": loss_ce.item(), "loss_align": loss_align.item()}


STEP_FNS = {
    "ce_only":    step_ce_only,
    "logit_kd":   step_logit_kd,
    "clean_align": step_clean_align,
    "adv_align":  step_adv_align,
    "traj_align": step_traj_align,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Architecture
    p.add_argument("--n_channels", type=int, default=256)
    p.add_argument("--expansion",  type=int, default=4)
    p.add_argument("--n_blocks",   type=int, default=6)

    # Teacher
    p.add_argument("--teacher", type=str, default="Wang2023Better_WRN-70-16",
                   help="RobustBench model name (CIFAR-10, Linf)")

    # Training
    p.add_argument("--mode",    type=str, default="adv_align", choices=list(STEP_FNS))
    p.add_argument("--epochs",  type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64,
                   help="adv_align uses double-backprop; 64 is safer for memory")
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_align", type=float, default=1.0)
    p.add_argument("--kd_temperature", type=float, default=4.0)

    # PGD
    p.add_argument("--eps",       type=float, default=8/255)
    p.add_argument("--alpha",     type=float, default=2/255)
    p.add_argument("--pgd_steps", type=int,   default=10)

    # Misc
    p.add_argument("--data_dir",   type=str, default="./data")
    p.add_argument("--save_dir",   type=str, default="./checkpoints")
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--eval_pgd_steps", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=10,
                   help="evaluate robust accuracy every N epochs (expensive)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, test_loader = get_cifar10_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Teacher (only needed for distillation modes)
    teacher = None
    if args.mode != "ce_only":
        print(f"Loading teacher: {args.teacher} ...")
        teacher = load_teacher(args.teacher, device)
        clean_acc = evaluate_clean(teacher, test_loader, device)
        print(f"Teacher clean accuracy: {clean_acc:.3%}")

    # Student
    student = CPLStudent(
        n_channels=args.n_channels, expansion=args.expansion, n_blocks=args.n_blocks
    ).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student params: {n_params/1e6:.2f}M")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Checkpoint dir
    save_dir = Path(args.save_dir) / args.mode
    save_dir.mkdir(parents=True, exist_ok=True)

    step_fn = STEP_FNS[args.mode]
    step_kwargs = dict(
        teacher=teacher,
        lambda_align=args.lambda_align,
        kd_temperature=args.kd_temperature,
        eps=args.eps,
        alpha=args.alpha,
        pgd_steps=args.pgd_steps,
    )

    best_clean = 0.0

    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_loss = 0.0
        t0 = time.time()

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss, _info = step_fn(student, x, y, **step_kwargs)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # -- Logging --
        dt = time.time() - t0
        clean_acc = evaluate_clean(student, test_loader, device)
        log_str = (f"[{epoch:3d}] loss={epoch_loss/len(train_loader):.4f}  "
                   f"clean={clean_acc:.3%}  lr={scheduler.get_last_lr()[0]:.2e}  {dt:.0f}s")

        if epoch % args.eval_every == 0:
            pgd_acc = evaluate_pgd(
                student, test_loader, device,
                eps=args.eps, alpha=args.alpha, n_steps=args.eval_pgd_steps,
            )
            log_str += f"  pgd20={pgd_acc:.3%}"

        print(log_str)

        # -- Save best --
        if clean_acc > best_clean:
            best_clean = clean_acc
            torch.save({
                "epoch": epoch,
                "model_state": student.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "clean_acc": clean_acc,
            }, save_dir / "best.pt")

        # -- Always save latest --
        torch.save({
            "epoch": epoch,
            "model_state": student.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
        }, save_dir / "latest.pt")

    print(f"\nDone. Best clean accuracy: {best_clean:.3%}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
