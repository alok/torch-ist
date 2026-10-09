# torch-ist

The **inverse scattering transform** in PyTorch: batched, device-agnostic, and differentiable.

The IST is the nonlinear analogue of the Fourier transform. Initial data is mapped to scattering
data (bound states, which are solitons, plus a reflection coefficient, which is radiation). Each component
evolves trivially in time, and the inverse transform (Gel'fand–Levitan–Marchenko) rebuilds the
solution. This package implements the full pipeline for

| PDE | Lax operator | solitons |
|---|---|---|
| KdV `u_t − 6uu_x + u_xxx = 0` | Schrödinger `−ψ'' + uψ = k²ψ` | `−2κ² sech²(κ(x − 4κ²t − x₀))` |
| focusing NLS `iq_t + q_xx + 2\|q\|²q = 0` | Zakharov–Shabat | `2η sech(2η(x − x₀ + 4ξt)) e^{−2iξx − 4i(ξ²−η²)t + iφ}` |

```bash
pip install -e .            # or: uv sync
pytest                      # 56 tests, ~40 s on CPU
python examples/kdv_soliton.py
```

## Quick start

```python
import torch, torch_ist as ti
torch.set_default_dtype(torch.float64)

x = torch.linspace(-20, 20, 801)
t = torch.linspace(0, 5, 101)

# KdV: u(x,0) -> scattering data -> evolve -> inverse
u, data = ti.kdv_ist_solve(-2 / torch.cosh(x)**2, x, t, reflection=False)
data.kappa, data.c2          # tensor([1.0000]), tensor([2.0000])

# general data (solitons + radiation): full Marchenko inversion
xs = torch.linspace(-20, 20, 321)
u, data = ti.kdv_ist_solve(-2.5 * torch.exp(-xs**2), xs, t[:31] * 0.3)

# NLS
q, data = ti.nls_ist_solve((2 / torch.cosh(x)).to(torch.complex128), x, t, reflection=False)
data.zeta                    # tensor([0+1.5j, 0+0.5j])  (Satsuma–Yajima breather)

# differentiable N-soliton solutions
kappa = torch.tensor([1.2, 0.7], requires_grad=True)
u = ti.kdv_from_solitons(x, t, kappa, x0=[-10.0, 0.0])
u.square().mean().backward()
```

## What is in the box

```
torch_ist/
  direct.py     direct transforms: Jost coefficients a, b via exact piecewise-constant
                transfer matrices (tree-reduced, O(log n) depth); bound states;
                norming constants; reflection coefficient; scattering-data dataclasses
                with .evolve(t) (t scalar or a tensor of times) and .to(device)
  inverse.py    closed-form N-soliton reconstruction (KdV and NLS), general Marchenko
                solvers with a scaled low-rank (Woodbury) soliton correction
  kdv.py        KdV pipeline, exact solitons, phase shifts, pseudo-spectral IF-RK4 solver
  nls.py        NLS pipeline, exact solitons, split-step Fourier solver
  utils.py      grids, spectral grids, 2x2 propagators
  visualize.py  spacetime heatmaps, IST-vs-PDE comparison panels, spectra
  neural.py     learnable IST: encoders, symmetry-structured decoder, exact latent evolution,
                hybrid exact/learned modes, data generation, losses, training, FNO baseline
```

### Numerics, briefly

* **Direct transform.** The potential is constant on each cell, so each cell's 2×2 propagator
  `cosh(μh)I + sinh(μh)/μ·A` is exact. The ordered product is reduced pairwise in O(log n) batched
  matmuls across every spectral parameter and batch element. The scheme is symmetric, so
  **Richardson extrapolation** against the every-other-sample grid gives O(h⁴) results.
* **Bound states.** KdV: `a(iκ)` is real, so a sign-change scan plus bisection finds every bound state.
  NLS: Fourier-collocation eigenvalues are refined by Newton's method on `a(ζ)`. Either way, a final
  **differentiable Newton step** gives exact implicit-function gradients `∂κ/∂u`.
* **Norming constants.** `b(iκₙ)` comes from matching the left and right Jost solutions *at that
  eigenfunction's peak*. For well-separated multi-solitons any fixed matching point fails: each
  eigenfunction is exponentially small in the other wells.
* **Inverse transform.** The soliton part of the Marchenko kernel is rank N but grows like
  `e^{2κ|x−xₙ|}`. Discretizing it naively loses all accuracy a few widths from each soliton. It is split off
  and handled with a **scaled Woodbury identity**, in which every matrix that is actually formed has O(1)
  entries. Its Gram integrals are exact, so only the radiation coupling is discretized. With `R = 0` this
  reduces to the classical determinant formula `u = −2∂²ₓ log det(I + A)`, which here never overflows.
  The derivation is in [docs/DERIVATIONS.md](docs/DERIVATIONS.md).
* **Spectral grids.** These are graded near k = 0, where the KdV reflection coefficient jumps to −1, and
  *time-aware*: radiation at wavenumber k sits near `s = 2x ≈ −24k²t` in `F(s)`, so
  `kdv_ist_solve` refines the grid so that the evolved kernel does not alias.

### Accuracy (float64, CPU)

| check | error |
|---|---|
| `−2 sech² x`: IST vs exact, t ∈ [0, 5] | 4e-6 |
| two-soliton phase shifts vs `log((κ₁+κ₂)/(κ₁−κ₂))/κ` | < 1e-4 |
| Pöschl–Teller eigenvalues `κ = ν, ν−1, …` | < 1e-6 |
| Marchenko round trip, `−3 sech² x` (2 solitons + radiation), h = 0.1 | 7e-3 |
| KdV `−2.5 e^{−x²}` (Marchenko) vs pseudo-spectral, t ≤ 1.5 | 1e-3 |
| NLS `2 sech x` breather vs split-step | 7e-4 |
| `∂κ/∂A` for `−A sech² x` (autograd) vs exact | < 1e-6 |

## Examples

Each example writes PNGs to `examples/output/` and asserts its own correctness checks.

| script | shows |
|---|---|
| `kdv_soliton.py` | `−2 sech² x` gives κ = 1, c² = 2, R ≈ 0; propagates at speed 4; heatmap and comparison with the PDE solver |
| `two_soliton_collision.py` | full pipeline on a sampled two-soliton profile; measured phase shifts vs theory |
| `nls_soliton.py` | `sech x` soliton, `2 sech x` breather, `1.2 sech x` soliton + radiation (Marchenko) |
| `kdv_radiation.py` | `−2.5 e^{−x²}`: soliton + dispersive tail, solitons-only part, spectrum |
| `learn_soliton_params.py` | autodiff: fit (κ, x₀) to a noisy snapshot; L-BFGS through the *direct* transform |
| `neural_ist.py` | train the scattering-space neural operator and an FNO baseline; generalization, speed, learned-law plots (~20 min on MPS; `--quick` for a smoke run) |

## Neural IST: a scattering-space neural operator (`torch_ist.neural`)

FNO learns in Fourier space, where *linear* PDEs diagonalize. For integrable PDEs, scattering space is the
diagonalizing domain: KdV time evolution there is `x₀ₙ += 4κₙ²t`, `R *= e^{8ik³t}`. `ScatteringNeuralOperator`
learns only the direct and inverse transforms and evolves **exactly** in between. Each transform can be set to
`"exact"` or `"learned"`, which gives a classical, hybrid, or fully neural model:

```python
from torch_ist.neural import ScatteringNeuralOperator, SolitonDetector, generate_kdv_soliton_data, train
x = torch.linspace(-20, 30, 384)
data = generate_kdv_soliton_data(6000, x.double())                 # exact (u0, scattering data) pairs
op = ScatteringNeuralOperator(x, n_slots=4, encoder=SolitonDetector(x, 4)).to("mps")
train(op, data, "encoder"); train(op, data, "decoder"); train(op, data, "end_to_end", physics_weight=1e-3)
op.set_mode(forward="learned", inverse="exact")                    # or any combination
u = op(u0, t)                                                       # (B, Nt, n)
```

Results from `examples/neural_ist.py` (relative L2 error of u(x,t); details in [docs/NEURAL.md](docs/NEURAL.md)):

| test set | learned inverse only | fully learned | FNO (same steps) |
|---|---|---|---|
| in distribution, t ≤ 2 | 1.3e-2 | 4.9e-2 | **1.1e-2** |
| **time extrapolation, 2 < t ≤ 4** | **4.6e-3** | **4.5e-2** | 1.49 |
| 4 solitons (trained on ≤ 3) | 0.14 | 0.63 | 0.13 |
| κ out of training range | **5.1e-2** | 0.49 | 0.43 |

* Time never enters a network, so extrapolating in t is free. FNO falls apart beyond its training horizon.
* The learned decoder rediscovers the KdV two-soliton phase-shift law `log((1+r)/|1−r|)` (error 0.08) and the
  soliton profile `−2sech²` (error 3e-4) without being told either.
* The learned *direct* transform is the weak link out of distribution (unseen amplitudes and counts).
  In distribution, FNO is 2–4× more accurate than the fully learned pipeline.
* Using the learned encoder in place of the exact direct transform makes inference 12× faster.

Includes data generation, scattering, potential and physics-informed (KdV residual) losses, a staged
`train()`, an FNO baseline, and a translation-equivariant, count-agnostic `SolitonDetector` encoder.

## GPU, batching, autodiff

* Every routine takes potentials of shape `(..., n)`. Scattering data are padded to the maximum
  soliton count with a `mask`, and padded solitons contribute exactly zero.
* All tensors follow the input's device, and `ScatteringData.to(device)` moves the data. Use float64 on CUDA.
  Apple MPS has no float64/complex128 support, so use CPU there.
* Gradients flow through: closed-form reconstruction (checked with `gradcheck`), time evolution,
  the direct transform (implicit eigenvalue derivatives), and the Marchenko solver.

## Conventions and caveats

* KdV: `u_t − 6uu_x + u_xxx = 0`. With the right reflection coefficient, `R(k,t) = R(k,0)e^{8ik³t}` and
  `cₙ²(t) = cₙ²(0)e^{8κₙ³t}`. Some references write `e^{4ik³t}`, which is for a different scaling of
  the equation.
* NLS: `ζ` is fixed, `Cₙ → Cₙe^{4iζₙ²t}`, and `r(ξ) → r(ξ)e^{4iξ²t}`.
* Grids must be uniform. The potential should be negligible near the domain edges.
* The Marchenko solver costs about 2n⁴ flops per time slice (n grid points). Use a few hundred points.
  Pure-soliton reconstruction is closed form and cheap at any resolution.
* Very shallow bound states (κ ≲ 10⁻³, or wider than the domain) are not detected.
* The reference PDE solvers are periodic. Fast KdV radiation (speed 12k²) wraps around unless the
  domain is padded, as `kdv_radiation.py` does.
