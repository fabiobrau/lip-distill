"""
PGD L-inf attack used both to craft adversarial examples for alignment
and for PGD-20 evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _random_init(x: torch.Tensor, eps: float) -> torch.Tensor:
    delta = torch.empty_like(x).uniform_(-eps, eps)
    return (x + delta).clamp(0.0, 1.0)


def pgd_linf(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    n_steps: int,
    random_start: bool = True,
    return_trajectory: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """
    PGD L-inf attack.

    Args:
        model:             target model (teacher or student).
        x:                 clean inputs in [0, 1], shape (B, C, H, W).
        y:                 ground-truth labels, shape (B,).
        eps:               L-inf perturbation budget (e.g. 8/255).
        alpha:             per-step size (e.g. 2/255).
        n_steps:           number of PGD steps.
        random_start:      start from a random perturbation inside the ball.
        return_trajectory: if True, also return the list of intermediate x_k.

    Returns:
        x_adv (always detached), and optionally the trajectory list.
    """
    model.eval()
    x_adv = _random_init(x, eps) if random_start else x.clone()
    trajectory = []

    for _ in range(n_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0.0, 1.0)
            if return_trajectory:
                trajectory.append(x_adv.clone())

    x_adv = x_adv.detach()
    if return_trajectory:
        return x_adv, trajectory
    return x_adv
