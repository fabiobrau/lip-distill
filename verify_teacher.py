#!/usr/bin/env python3
"""
Milestone 1: Verify teacher clean + robust accuracy against RobustBench leaderboard.

Wang2023Better_WRN-70-16 (CIFAR-10, Linf, eps=8/255):
  Clean accuracy:          93.25 %
  AutoAttack robust acc:   75.22 %

Usage:
  python verify_teacher.py                      # PGD-20 only (fast)
  python verify_teacher.py --autoattack         # + AutoAttack (slow, ~30 min)
  python verify_teacher.py --aa_samples 500     # AutoAttack on 500 samples
"""

import argparse

import torch
from tqdm import tqdm

from data import get_cifar10_loaders
from attacks import pgd_linf

LEADERBOARD = {
    "Wang2023Better_WRN-70-16": {"clean": 93.25, "autoattack": 75.22},
    "Wang2023Better_WRN-28-10": {"clean": 92.44, "autoattack": 67.68},
}


def load_teacher(model_name: str, device: torch.device):
    from robustbench.utils import load_model
    teacher = load_model(model_name=model_name, dataset="cifar10", threat_model="Linf")
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def eval_clean(model, loader, device):
    correct = total = 0
    for x, y in tqdm(loader, desc="Clean"):
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def eval_pgd(model, loader, device, eps, alpha, n_steps, n_batches=None):
    correct = total = 0
    for i, (x, y) in enumerate(tqdm(loader, desc=f"PGD-{n_steps}")):
        if n_batches is not None and i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        x_adv = pgd_linf(model, x, y, eps=eps, alpha=alpha, n_steps=n_steps)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def eval_autoattack(model, loader, device, eps, n_samples):
    try:
        from autoattack import AutoAttack
    except ImportError:
        print("  autoattack not installed. Run: pip install autoattack")
        return None

    xs, ys = [], []
    for x, y in loader:
        xs.append(x); ys.append(y)
        if sum(v.size(0) for v in xs) >= n_samples:
            break
    x_all = torch.cat(xs)[:n_samples].to(device)
    y_all = torch.cat(ys)[:n_samples].to(device)

    adversary = AutoAttack(model, norm="Linf", eps=eps, version="standard", device=device)
    x_adv = adversary.run_standard_evaluation(x_all, y_all, bs=64)
    with torch.no_grad():
        acc = (model(x_adv).argmax(1) == y_all).float().mean().item()
    return 100.0 * acc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher",     type=str,   default="Wang2023Better_WRN-70-16")
    p.add_argument("--data_dir",    type=str,   default="./data")
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--eps",         type=float, default=8/255)
    p.add_argument("--alpha",       type=float, default=2/255)
    p.add_argument("--pgd_steps",   type=int,   default=20)
    p.add_argument("--pgd_batches", type=int,   default=None,
                   help="Limit PGD eval to N batches (None = full test set)")
    p.add_argument("--autoattack",  action="store_true")
    p.add_argument("--aa_samples",  type=int,   default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Teacher: {args.teacher}\n")

    _, test_loader = get_cifar10_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    print("Loading teacher from RobustBench ...")
    teacher = load_teacher(args.teacher, device)

    # --- Clean accuracy ---
    clean = eval_clean(teacher, test_loader, device)
    ref_clean = LEADERBOARD.get(args.teacher, {}).get("clean")
    clean_ok = ref_clean is None or abs(clean - ref_clean) < 0.5
    print(f"\nClean accuracy:  {clean:.2f}%  "
          f"(leaderboard: {ref_clean}%)  {'OK' if clean_ok else 'MISMATCH'}")

    # --- PGD robust accuracy ---
    pgd = eval_pgd(teacher, test_loader, device,
                   eps=args.eps, alpha=args.alpha, n_steps=args.pgd_steps,
                   n_batches=args.pgd_batches)
    ref_aa = LEADERBOARD.get(args.teacher, {}).get("autoattack")
    print(f"PGD-{args.pgd_steps} accuracy: {pgd:.2f}%  "
          f"(AA leaderboard: {ref_aa}%  — PGD is an upper bound on true robust acc)")

    # --- AutoAttack (optional) ---
    if args.autoattack:
        print(f"\nRunning AutoAttack on {args.aa_samples} samples ...")
        aa = eval_autoattack(teacher, test_loader, device,
                             eps=args.eps, n_samples=args.aa_samples)
        if aa is not None:
            aa_ok = ref_aa is None or abs(aa - ref_aa) < 1.0
            print(f"AutoAttack accuracy: {aa:.2f}%  "
                  f"(leaderboard: {ref_aa}%)  {'OK' if aa_ok else 'MISMATCH'}")

    # --- Summary ---
    print("\n=== Summary ===")
    status = "PASS" if clean_ok else "FAIL (clean accuracy mismatch)"
    print(f"Teacher verification: {status}")
    if not clean_ok:
        print("  Check that the correct model name / dataset / threat_model are used.")


if __name__ == "__main__":
    main()
