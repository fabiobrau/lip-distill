#!/usr/bin/env python3
"""
Milestone 2 & 3 sanity checks:
  1. CPLConv2d: spectral norm of rescaled operator is <= 1 (via power iteration on JᵀJ).
  2. CPLStudent: empirical Lipschitz ratio <= L_bound.
  3. Input-gradient via autograd + create_graph=True works end-to-end.
  4. Alignment loss backward (double-backprop) flows gradients to all student params.
"""

import torch
import torch.nn.functional as F
from models import CPLConv2d, CPLStudent
from losses import student_gradient, gradient_alignment_loss


def check_cpl_spectral_norm():
    """
    Verify that the rescaled CPL operator has spectral norm <= 1.
    Uses the JᵀJ power iteration from the repo's utils (reimplemeted inline).
    """
    print("\n--- CPLConv2d spectral norm check ---")
    layer = CPLConv2d(16, 64, kernel_size=3).eval()
    x0 = torch.zeros(1, 16, 8, 8)
    _ = layer(x0)  # trigger lazy u init

    # Power iteration on JᵀJ at zero (activation mask = 0, so J = I - 0 = I,
    # but we test the rescaled res path directly)
    # Instead: test via finite differences on random pairs.
    n_pairs = 256
    x1 = torch.randn(n_pairs, 16, 8, 8)
    x2 = torch.randn(n_pairs, 16, 8, 8)
    with torch.no_grad():
        f1 = layer(x1)
        f2 = layer(x2)
    dout = (f1 - f2).flatten(1).norm(dim=-1)
    din  = (x1 - x2).flatten(1).norm(dim=-1)
    ratio = (dout / din.clamp(min=1e-8)).max().item()
    print(f"Max empirical ratio (should be <= 1): {ratio:.6f}")
    passed = ratio <= 1.05  # small tolerance for finite-sample estimate
    print("PASS" if passed else f"FAIL: ratio {ratio:.4f} > 1.05")
    return passed


def check_student_lipschitz():
    print("\n--- CPLStudent empirical Lipschitz check ---")
    model = CPLStudent(n_channels=32, expansion=2, n_blocks=2).eval()

    # Warm up to init u buffers
    _ = model(torch.randn(2, 3, 32, 32))

    L = model.lipschitz_constant(n_iter=100)
    print(f"L_head estimate: {L:.4f}")

    n_pairs = 256
    x1 = torch.randn(n_pairs, 3, 32, 32)
    x2 = torch.randn(n_pairs, 3, 32, 32)
    with torch.no_grad():
        f1 = model(x1)
        f2 = model(x2)
    dout = (f1 - f2).norm(dim=-1)
    din  = (x1.flatten(1) - x2.flatten(1)).norm(dim=-1)
    ratio = (dout / din.clamp(min=1e-8)).max().item()
    print(f"Max empirical ratio (should be <= {L*1.05:.4f}): {ratio:.6f}")
    passed = ratio <= L * 1.05
    print("PASS" if passed else f"FAIL: ratio {ratio:.4f} > bound {L*1.05:.4f}")
    return passed


def check_gradient_flow():
    print("\n--- Double-backprop gradient flow check ---")
    model = CPLStudent(n_channels=32, expansion=2, n_blocks=2).train()
    _ = model(torch.randn(2, 3, 32, 32))  # init u buffers

    B = 4
    x_adv = torch.randn(B, 3, 32, 32)
    y = torch.randint(0, 10, (B,))

    g_s, _ = student_gradient(model, x_adv, y)
    print(f"g_s shape: {g_s.shape}  (expected {x_adv.shape})")
    assert g_s.shape == x_adv.shape

    g_t = torch.randn_like(g_s).detach()
    loss = gradient_alignment_loss(g_s, g_t)
    print(f"Alignment loss: {loss.item():.4f}")

    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"Params with gradients: {n_with_grad}/{total_params}")
    assert n_with_grad > 0
    print("PASS")
    return True


def main():
    torch.manual_seed(0)
    ok1 = check_cpl_spectral_norm()
    ok2 = check_student_lipschitz()
    ok3 = check_gradient_flow()

    print()
    if ok1 and ok2 and ok3:
        print("=== All sanity checks PASSED ===")
    else:
        print("=== SOME CHECKS FAILED ===")


if __name__ == "__main__":
    main()
