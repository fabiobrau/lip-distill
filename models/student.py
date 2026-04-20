"""
CPL student network for CIFAR-10.

Architecture:
  SN-Conv2d(3 → channels)
  → [CPLConv2d(channels, channels*expansion, kernel_size)] × n_blocks
  → AdaptiveAvgPool2d(1)
  → Linear(channels, n_classes)

Lipschitz constant: L_proj × 1^n_blocks × L_head
  L_proj = ||W_proj||_2   (spectral-norm constrained ≤ 1 by nn.utils.spectral_norm)
  1^n_blocks               each CPLConv2d is 1-Lipschitz
  L_head = ||W_head||_2   (unconstrained; computed by power iteration for certification)

Overall L ≤ L_head.
Certified L2-radius: r = (f_1(x) - f_2(x)) / (2 · L).
"""

import torch
import torch.nn as nn

from .cpl import CPLConv2d, spectral_norm_estimate


class CPLStudent(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_channels: int = 256,
        expansion: int = 4,       # hidden dim inside CPLConv2d = n_channels * expansion
        n_blocks: int = 6,
        kernel_size: int = 3,
        n_classes: int = 10,
    ):
        super().__init__()

        # Initial projection: spectrally normalised so ||W_proj||_2 ≤ 1
        self.proj = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, n_channels, kernel_size, padding=kernel_size // 2, bias=True)
        )

        # Stack of 1-Lipschitz CPL blocks (all operate at the same spatial resolution)
        self.blocks = nn.Sequential(*[
            CPLConv2d(n_channels, n_channels * expansion, kernel_size)
            for _ in range(n_blocks)
        ])

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Head: unconstrained; contributes L_head to overall Lipschitz constant
        self.head = nn.Linear(n_channels, n_classes)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)       # (B, n_channels, H, W)
        x = self.blocks(x)     # (B, n_channels, H, W)
        x = self.pool(x)       # (B, n_channels, 1, 1)
        x = x.flatten(1)       # (B, n_channels)
        return self.head(x)    # (B, n_classes)

    @torch.no_grad()
    def lipschitz_constant(self, n_iter: int = 50) -> float:
        """
        Upper bound: L_proj ≤ 1 (spectral_norm), CPL blocks = 1, L_head = ||W_head||_2.
        Returns L_head (the dominant factor).
        """
        return spectral_norm_estimate(self.head.weight, n_iter).item()

    @torch.no_grad()
    def certified_radii(self, x: torch.Tensor, L: float | None = None) -> torch.Tensor:
        """Per-sample certified L2-radius: r = (f_1 - f_2) / (2·L)."""
        if L is None:
            L = self.lipschitz_constant()
        logits = self.forward(x)
        top2 = logits.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).clamp(min=0.0)
        return margin / (2.0 * L)
