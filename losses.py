"""
Loss functions for gradient-alignment distillation.

gradient_alignment_loss : 1 - cosine(g_student, g_teacher)
logit_kd_loss           : KL divergence between teacher and student softmax
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _input_gradient(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    """
    Gradient of cross-entropy loss w.r.t. x.
    x must already have requires_grad=True.
    create_graph=True retains the graph for double-backprop (student only).
    """
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    (grad,) = torch.autograd.grad(loss, x, create_graph=create_graph)
    return grad


def teacher_gradient(
    teacher: nn.Module,
    x_adv: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Input gradient of the frozen teacher at x_adv. Returned detached.
    """
    x_t = x_adv.detach().requires_grad_(True)
    g_t = _input_gradient(teacher, x_t, y, create_graph=False)
    return g_t.detach()


def student_gradient(
    student: nn.Module,
    x_adv: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Input gradient of the student at x_adv, with graph retained for
    double-backprop.  Also returns the student logits for reuse.

    Returns:
        g_s  : gradient (B, *input_shape), graph attached.
        logits_adv : student logits at x_adv (graph attached).
    """
    x_s = x_adv.detach().requires_grad_(True)
    logits_adv = student(x_s)
    loss = F.cross_entropy(logits_adv, y)
    (g_s,) = torch.autograd.grad(loss, x_s, create_graph=True)
    return g_s, logits_adv


def gradient_alignment_loss(
    g_s: torch.Tensor,
    g_t: torch.Tensor,
    min_teacher_norm: float = 1e-6,
) -> torch.Tensor:
    """
    L_align = mean over batch of  (1 - cos(g_s, g_t)),
    skipping samples where teacher gradient is near-zero (flat regions).
    """
    B = g_s.shape[0]
    gs_flat = g_s.reshape(B, -1)
    gt_flat = g_t.reshape(B, -1)

    # Mask out flat regions: teacher gradient is unreliable there
    mask = gt_flat.norm(dim=-1) > min_teacher_norm
    if mask.sum() == 0:
        return g_s.sum() * 0.0  # differentiable zero

    cos = F.cosine_similarity(gs_flat[mask], gt_flat[mask], dim=-1)
    return (1.0 - cos).mean()


def logit_kd_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """KL-divergence knowledge distillation (Hinton et al.)."""
    T = temperature
    p_t = F.softmax(logits_t / T, dim=-1)
    log_p_s = F.log_softmax(logits_s / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T ** 2)
