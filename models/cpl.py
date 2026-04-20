"""
CPLConv2d: convolutional Convex Potential Layer.
Faithful port of berndprach/1LipschitzLayersCompared/lipnn/layers/lipschitz/cpl.py.

Forward: y = x - (2/σ²) * Wᵀ * ReLU(W*x + b)
where σ = spectral norm of the conv operator.

1-Lipschitz proof:
  Jacobian J = I - τ·Wᵀ D W,  τ = 2/σ²,  D = activation mask (PSD, 0/1 diagonal).
  ||τ Wᵀ D W||₂ ≤ τ·σ² = 2  →  eigenvalues of J in [-1, 1]  →  ||J||₂ ≤ 1.

Reference: Meunier et al., "A Dynamical System Perspective for Lipschitz NNs" (2022).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CPLConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,      # hidden expansion dim; output has in_channels channels
        kernel_size: int = 3,
        padding: str | int = "same",
        val_niter: int = 100,
    ) -> None:
        super().__init__()
        self.epsilon = 1e-6
        self.val_niter = val_niter
        self.padding = padding

        self.kernel = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.u_initialized = False
        self.val_rescaling_cached = None

    # ------------------------------------------------------------------
    # Power iteration (operates in feature-map space, same as repo)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_uv(self, num_iter: int, epsilon: float):
        u = self.u
        tp_pad = (self.kernel.shape[-1] - 1) // 2
        for _ in range(num_iter):
            v0 = F.conv2d(u, self.kernel, padding="same")
            v  = v0 / (v0.norm(p=2) + epsilon)
            u0 = F.conv_transpose2d(v, self.kernel, padding=tp_pad)
            u  = u0 / (u0.norm(p=2) + epsilon)
        return u, v

    def _rescaling(self, u, v) -> torch.Tensor:
        conv_u = F.conv2d(u, self.kernel, padding="same")
        return 2.0 / ((conv_u * v).sum() ** 2 + self.epsilon)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_pad = (self.kernel.shape[-1] - 1) // 2

        res = F.conv2d(x, self.kernel, bias=self.bias, padding=self.padding)
        res = F.relu(res)
        res = F.conv_transpose2d(res, self.kernel, padding=tp_pad)

        # Lazy initialisation of u (spatial dims not known at __init__ time).
        # u lives in the INPUT space (in_channels), not the hidden space.
        # The repo uses kernel.shape[0] (out_channels) which is a bug that only
        # manifests when out_channels != in_channels (i.e. when expansion > 1).
        if not self.u_initialized:
            u_size = (1, self.kernel.shape[1], x.shape[-2], x.shape[-1])  # in_channels
            self.register_buffer("u", torch.randn(u_size, device=x.device))
            self.u_initialized = True

        if self.training:
            self.val_rescaling_cached = None
            u, v = self._update_uv(1, self.epsilon)
            self.u = u
            rescaling = self._rescaling(u, v)
        elif self.val_rescaling_cached is None:
            u, v = self._update_uv(self.val_niter, 1e-12)
            rescaling = self._rescaling(u, v)
            self.val_rescaling_cached = rescaling
        else:
            rescaling = self.val_rescaling_cached

        return x - rescaling * res

    # ------------------------------------------------------------------
    # State dict: handle dynamically-registered buffer 'u'
    # ------------------------------------------------------------------
    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + "u" in state_dict:
            u = state_dict.pop(prefix + "u").to(self.kernel.device)
            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys, unexpected_keys, error_msgs)
            self.register_buffer("u", u)
            self.u_initialized = True
        else:
            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys, unexpected_keys, error_msgs)

    def extra_repr(self) -> str:
        k = self.kernel
        return (f"in_channels={k.shape[1]}, out_channels={k.shape[0]}, "
                f"kernel_size={tuple(k.shape[-2:])}")


# ---------------------------------------------------------------------------
# Utility: spectral norm estimate via power iteration (for head / projection)
# ---------------------------------------------------------------------------

def spectral_norm_estimate(W: torch.Tensor, n_iter: int = 20) -> torch.Tensor:
    u = F.normalize(torch.randn(W.shape[0], device=W.device, dtype=W.dtype), dim=0)
    with torch.no_grad():
        for _ in range(n_iter):
            v = F.normalize(W.t() @ u, dim=0)
            u = F.normalize(W @ v, dim=0)
    return (u @ W @ v).abs()
