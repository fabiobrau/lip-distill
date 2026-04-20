#!/usr/bin/env python3
"""
Evaluation script for CPL student.

Metrics:
  - Clean accuracy
  - PGD-20 robust accuracy (fast approximation)
  - AutoAttack robust accuracy (tight, slow)
  - Certified L2-radius stats (Lipschitz certificate)

Example:
  python evaluate.py --checkpoint checkpoints/adv_align/best.pt
  python evaluate.py --checkpoint checkpoints/adv_align/best.pt --autoattack
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import get_cifar10_loaders
from models import CPLStudent
from attacks import pgd_linf


def load_student(ckpt_path: str, device: torch.device) -> tuple[CPLStudent, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    student = CPLStudent(
        n_channels=args.get("n_channels", 256),
        expansion=args.get("expansion", 4),
        n_blocks=args.get("n_blocks", 6),
    ).to(device)
    student.load_state_dict(ckpt["model_state"])
    student.eval()
    return student, args


@torch.no_grad()
def eval_clean(student, loader, device):
    correct = total = 0
    for x, y in tqdm(loader, desc="Clean"):
        x, y = x.to(device), y.to(device)
        correct += (student(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def eval_pgd(student, loader, device, eps, alpha, n_steps=20):
    correct = total = 0
    for x, y in tqdm(loader, desc=f"PGD-{n_steps}"):
        x, y = x.to(device), y.to(device)
        x_adv = pgd_linf(student, x, y, eps=eps, alpha=alpha, n_steps=n_steps)
        with torch.no_grad():
            correct += (student(x_adv).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def eval_autoattack(student, loader, device, eps, n_samples=1000):
    try:
        from autoattack import AutoAttack
    except ImportError:
        print("autoattack not installed. Run: pip install autoattack")
        return None

    # Collect first n_samples
    xs, ys = [], []
    for x, y in loader:
        xs.append(x); ys.append(y)
        if sum(v.size(0) for v in xs) >= n_samples:
            break
    x_all = torch.cat(xs)[:n_samples].to(device)
    y_all = torch.cat(ys)[:n_samples].to(device)

    adversary = AutoAttack(student, norm="Linf", eps=eps, version="standard", device=device)
    with torch.no_grad():
        x_adv = adversary.run_standard_evaluation(x_all, y_all, bs=64)
        correct = (student(x_adv).argmax(1) == y_all).float().mean().item()
    return correct


@torch.no_grad()
def eval_certified(student, loader, device, eps_linf=8/255):
    """
    Certified L2-robustness via Lipschitz bound.
    Lipschitz constant = ||W_head||_2  (backbone is 1-Lip, projection is 1-Lip).
    Certified radius r = (f_1 - f_2) / (2L).
    Reports: mean/median certified radius, fraction certified at eps (L2 conversion).
    """
    L = student.lipschitz_constant()
    print(f"  Lipschitz constant (||W_head||_2): {L:.4f}")

    all_radii = []
    all_correct = []

    for x, y in tqdm(loader, desc="Certified"):
        x, y = x.to(device), y.to(device)
        logits = student(x)
        preds = logits.argmax(1)
        top2 = logits.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).clamp(min=0.0)
        radii = (margin / (2.0 * L)).cpu().numpy()
        all_radii.append(radii)
        all_correct.append((preds == y).cpu().numpy())

    all_radii = np.concatenate(all_radii)
    all_correct = np.concatenate(all_correct)

    # L_inf eps -> L2 equivalent for 32x32x3 image
    eps_l2_equiv = eps_linf * (32 * 32 * 3) ** 0.5
    certified = all_correct & (all_radii >= eps_l2_equiv)

    print(f"  Certified accuracy at eps_L2={eps_l2_equiv:.3f} (equiv to Linf {eps_linf:.4f}): "
          f"{certified.mean():.3%}")
    print(f"  Mean certified radius: {all_radii.mean():.4f}")
    print(f"  Median certified radius: {np.median(all_radii):.4f}")
    return all_radii, certified


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir",   type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--eps",        type=float, default=8/255)
    p.add_argument("--alpha",      type=float, default=2/255)
    p.add_argument("--pgd_steps",  type=int,   default=20)
    p.add_argument("--autoattack", action="store_true",
                   help="Run AutoAttack (slow but tight bound)")
    p.add_argument("--aa_samples", type=int, default=1000)
    p.add_argument("--skip_pgd",   action="store_true")
    p.add_argument("--certified",  action="store_true",
                   help="Compute Lipschitz certified radius")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    student, train_args = load_student(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  mode={train_args.get('mode')}, "
          f"hidden_dim={train_args.get('hidden_dim')}, "
          f"n_blocks={train_args.get('n_blocks')}")

    _, test_loader = get_cifar10_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    clean = eval_clean(student, test_loader, device)
    print(f"\nClean accuracy:    {clean:.3%}")

    if not args.skip_pgd:
        pgd = eval_pgd(student, test_loader, device,
                       eps=args.eps, alpha=args.alpha, n_steps=args.pgd_steps)
        print(f"PGD-{args.pgd_steps} accuracy: {pgd:.3%}")

    if args.autoattack:
        aa = eval_autoattack(student, test_loader, device,
                             eps=args.eps, n_samples=args.aa_samples)
        if aa is not None:
            print(f"AutoAttack accuracy: {aa:.3%}")

    if args.certified:
        print("\nCertified robustness:")
        eval_certified(student, test_loader, device, eps_linf=args.eps)


if __name__ == "__main__":
    main()
