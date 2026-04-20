# Gradient-Alignment Distillation: ResNet → Lipschitz (CPL/SLL) Student

## Thesis

Lipschitz constraints bound the **magnitude** of the input-gradient ($\|\nabla_x f\|_2 \le L$), but adversarial robustness depends on its **direction**. Transfer the direction from an adversarially-trained (non-Lipschitz) teacher into a Lipschitz-by-construction student via cosine alignment of input-gradients.

- **Teacher:** any adversarially robust model — recommended to pull a pretrained one from **[RobustBench](https://robustbench.github.io/)** (e.g. a ResNet trained with PGD / TRADES / MART on CIFAR-10 at $\epsilon=8/255$). Architecture need not be a ResNet; any differentiable robust classifier works. On-the-fly training is an option but not required (see note below).
- **Student:** stack of residual Lipschitz blocks of the form
  $$y = x - 2 W^\top \sigma(Wx + b), \quad \sigma = \text{ReLU}$$
  with $\|W\|_2 \le 1$. Two interchangeable variants:
  - **SLL** (SDP-based Lipschitz Layer, Araujo et al. 2023) with AOL-style reparameterization.
  - **CPL** (Convex Potential Layer, Meunier et al. 2022) — same functional form, slightly different parameterization/normalization of $W$.

  Both give 1-Lipschitz blocks and the closed-form Jacobian below is identical. **Start with CPL** (simpler, well-tested in the Lipschitz-robustness literature); swap to SLL if you want tighter bounds or better expressivity.
- **Transfer signal:** cosine similarity between $\nabla_x f_s$ and $\nabla_x f_t$, evaluated at **adversarial points** (not clean points).

---

## Explicit Form

### CPL block Jacobian (per layer)

$$J_\ell(x) = I - 2 W_\ell^\top D_\ell(x) W_\ell, \quad D_\ell(x) = \text{diag}(\mathbb{1}[W_\ell x + b_\ell > 0])$$

- 1-Lipschitz when $\|W_\ell\|_2 \le 1$ (reflection-like operator).
- $D_\ell$ is piecewise constant in $x$ at fixed $W$ — a theoretical property (Jacobian is locally affine within each activation cell), **not** an implementation shortcut. Masks must be recomputed at every new input ($x$ vs $x+\delta$) and every training step (weights change each step).

### Student input-gradient for logit $k$

$$g_s^{(k)}(x) = \prod_{\ell=1}^{L} J_\ell(x)^\top \cdot W_{\text{head}}^\top e_k$$

### Gradient of alignment loss w.r.t. $W_\ell$

With $P_\ell = I - 2 W_\ell^\top D_\ell W_\ell$, $A_\ell = \prod_{j<\ell} P_j^\top$, $B_\ell = \prod_{j>\ell} P_j^\top$, teacher gradient $g_t$ (stop-grad), and $\hat g = g / \|g\|$:

$$\nabla_{W_\ell} \mathcal{L}_{\text{align}} = -\frac{1}{\|g_s\|} \big(I - \hat g_s \hat g_s^\top\big) \hat g_t \cdot \frac{\partial g_s}{\partial W_\ell}$$

$$\frac{\partial g_s^{(k)}}{\partial W_\ell} = -2 A_\ell \big(D_\ell W_\ell (\cdot) + (\cdot) W_\ell^\top D_\ell\big) B_\ell e_k \quad \text{(symmetrized, bilinear in } W_\ell\text{)}$$

### Status of the closed-form: theoretical, not default implementation

The bilinear-in-$W_\ell$ structure above is useful for **understanding** (the alignment update has a specific algebraic form) and as a potential **optimization** if double-backprop becomes the bottleneck. It is **not** the recommended first implementation path.

**Default implementation:** use PyTorch autograd.
```python
g_s = torch.autograd.grad(f_s(x_adv).sum(dim=0), x_adv, create_graph=True)[0]
# ... compute alignment loss against g_t (detached) ...
# loss.backward() handles double-backprop automatically
```
This is how Jacobian-regularization papers (Hoffman et al., Chan et al.) actually implement gradient-based losses. Expensive but tractable. Profile first; hand-roll the closed-form only if justified.

**AOL caveat (SLL variant only):** AOL reparameterizes $W = \tilde W / \sqrt{\text{diag}(|\tilde W^\top \tilde W|\mathbf{1})}$. Autograd handles the chain rule automatically; only relevant if you hand-roll.

---

## Key Design Choice: Align at Adversarial Points

### Why not clean points

Robustness is a neighborhood property. Clean-point alignment samples the teacher's gradient field on a measure-zero slice (the data manifold). The robustness-relevant information lives on the $\epsilon$-ball boundary where the teacher was trained.

### Formulation

Let $\delta_t(x) = \arg\max_{\|\delta\|\le\epsilon} \mathcal{L}_{\text{CE}}(f_t(x+\delta), y)$ (PGD against teacher).

$$\mathcal{L}_{\text{align}}^{\text{adv}}(x) = 1 - \cos\big(\nabla_x f_s(x + \delta_t(x)),\ \nabla_x f_t(x + \delta_t(x))\big)$$

### Compatibility with the explicit form

Moving the evaluation point from $x$ to $x+\delta$ does not change the structure of the derivation: one student forward pass on $x+\delta$ produces masks $D_\ell(x+\delta)$ and the same Jacobian product formula applies. Masks at $x$ and at $x+\delta$ are **independent** — both are computed per forward pass, no reuse between them.

### Attack source options

1. **Teacher attack** $\delta_t$: transfers teacher's robustness geometry. Good curriculum early in training.
2. **Student attack** $\delta_s$: addresses student's own weaknesses. Better late in training.
3. **Shared attack**: hybrid.

**Recommended:** start with (1), anneal toward (2).

### Trajectory alignment (cheap extension)

PGD produces $x \to x_1 \to \ldots \to x_K$ for free. Align along full trajectory:

$$\mathcal{L}_{\text{align}}^{\text{traj}} = \frac{1}{K}\sum_{k=1}^K \big(1 - \cos(\nabla_x f_s(x_k), \nabla_x f_t(x_k))\big)$$

Denser supervision, no extra forward cost.

---

## Teacher: Frozen (RobustBench) vs On-the-Fly

**Default: frozen RobustBench teacher.** Simplest path; teacher gradient $g_t$ is a stable target.

**Optional: on-the-fly teacher.** Teacher non-stationary if co-trained. Two options:

1. **Stop-grad teacher per step** (recommended if going on-the-fly). Two-timescale SGD; stable if teacher updates slower than student. Add EMA on target gradient:
   $$\bar g_t \leftarrow \alpha \bar g_t + (1-\alpha) g_t, \quad \alpha \approx 0.9\text{–}0.99$$
2. Joint bilevel optimization — requires third-order terms, not worth it.

Run frozen first. If frozen-teacher transfer works, on-the-fly is a secondary ablation (does a co-adapting teacher shape better transfer?). If frozen fails, on-the-fly won't rescue it.

---

## Practical Notes

- **Magnitude mismatch:** $\|\nabla_x f_t\| \gg \|\nabla_x f_s\| \le L_s$. Cosine ignores this, but downweight samples where $\|\nabla_x f_t(x+\delta)\|$ is near zero (flat regions → noisy direction).
- **Total loss:**
  $$\mathcal{L} = \mathcal{L}_{\text{CE}}(f_s(x), y) + \lambda \mathcal{L}_{\text{align}}^{\text{adv}}(x) \;(+ \text{optional KD on logits})$$
- **Baselines to run first:**
  1. Student trained with CE only (Lipschitz lower bound).
  2. Student with logit KD from teacher (standard distillation).
  3. Student with clean-point gradient alignment (ablation for "adv-point matters").
  4. Full method (adv-point alignment).
- **Frozen vs on-the-fly teacher:** run frozen-pretrained teacher first. If that fails, on-the-fly won't rescue it.

---

## Open Questions to Validate

1. Does gradient *direction* alone (with Lipschitz magnitude control) suffice for robustness? ~60/40 optimistic.
2. Does on-the-fly teacher help vs frozen pretrained? Unknown — needs ablation.
3. Does trajectory alignment beat endpoint-only alignment?

---

## Prior Art to Check

- **RobustBench** (Croce et al. 2021) — standardized robust model zoo; source of teacher.
- Sarkar et al., *Get Fooled for the Right Reason* (gradient alignment).
- Chan et al., *Jacobian Adversarially Regularized Networks*.
- Ilyas et al. 2019, Engstrom et al. 2019 (robust features / perceptually aligned gradients).
- Araujo et al. 2023, *A Unified Algebraic Perspective on Lipschitz NNs* (SLL).
- Meunier et al. 2022, *A Dynamical System Perspective for Lipschitz Neural Networks* (CPL).
- AOL: Prach & Lampert, *Almost-Orthogonal Layers*.

---

## Architecture Starting Point

- **Teacher:** pretrained model from RobustBench (CIFAR-10, $\ell_\infty$, $\epsilon=8/255$). Suggested: a standard ResNet-18 or WideResNet-28-10 entry from the leaderboard. Load via the `robustbench` Python package:
  ```python
  from robustbench.utils import load_model

  teacher = load_model(model_name="...", dataset="cifar10", threat_model="Linf")
  ```
- **Student:** 4–8 CPL blocks + linear head, $\|W_\ell\|_2 \le 1$, ~matching parameter count where feasible. Take the CPL from https://github.com/berndprach/1LipschitzLayersCompared/tree/main/lipnn if you prefer it.
- **Optimizer:** AdamW, cosine LR schedule.
- **Alignment:** PGD-10 on teacher, align at endpoint (phase 1), then trajectory (phase 2).
- **Eval:** clean accuracy, AutoAttack $\ell_\infty$ at $\epsilon=8/255$, certified robustness via Lipschitz bound.

---

## First Milestones

1. Load a RobustBench teacher; sanity-check its clean + robust accuracy matches leaderboard numbers.
2. Implement the CPL block or download it; If implement, verify 1-Lipschitz numerically via power iteration **on the full stacked network** (not just per-layer).
3. Implement input-gradient computation via `torch.autograd.grad(..., create_graph=True)`; verify gradient flow end-to-end on a small batch.
4. Frozen-teacher baseline with clean-point alignment → sanity check.
5. Switch to adv-point alignment → expect robustness gain.
6. Add trajectory alignment → ablate.
7. (Optional, only if required) On-the-fly teacher with EMA targets → secondary ablation.