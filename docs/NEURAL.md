# Scattering-space neural operators

## Idea

A Fourier neural operator (FNO) learns in the basis where *linear* constant-coefficient PDEs
diagonalize. In that basis a time step is pointwise multiplication, `û(k,t) = e^{iω(k)t} û(k,0)`.
FNO keeps the Fourier transform fixed and learns a nonlinear map in between.

For integrable nonlinear PDEs, the inverse scattering transform (IST) is the diagonalizing map.
For KdV (`u_t − 6uu_x + u_xxx = 0`) in the coordinates used here:

| scattering coordinate | evolution |
|---|---|
| eigenvalue κₙ | fixed |
| soliton position x₀ₙ (from cₙ² = 2κₙe^{2κₙx₀ₙ}) | `x₀ₙ + 4κₙ²t` (translation) |
| reflection coefficient R(k) | `R(k)e^{8ik³t}` (phase rotation) |

`ScatteringNeuralOperator` keeps that evolution **exact** and learns only the two transforms:

```
u(x,0) ──encoder E_θ──▶ (κ, x₀, p, R) ──exact flow Φ_t──▶ (κ, x₀ + 4κ²t, p, Re^{8ik³t}) ──decoder D_φ──▶ u(x,t)
```

Either transform can be swapped for the exact one at run time, which gives four models:

| forward / inverse | what it is |
|---|---|
| exact / exact | classical IST (reference; float64, CPU) |
| exact / learned | learned inverse transform. The fast amortized replacement for Marchenko inversion. |
| learned / exact | learned direct transform. Spectral estimation from data without eigen-solves. |
| learned / learned | fully neural, but time enters *only* through the exact flow |

### Why this should generalize where FNO does not

* **Time extrapolation.** FNO receives `t` as an input feature and has to learn the dynamics. Outside the
  training range of `t` it is extrapolating a learned function. Here `t` never enters a network. A larger
  `t` only produces larger soliton separations, and the decoder handles those because it is
  translation-equivariant (below). Extrapolation in time therefore reduces to interpolation in configuration.
* **Conserved quantities for free.** `κ` is never touched after encoding, so mass, momentum, energy, and every
  higher KdV invariant (all functions of κₙ and R) are conserved exactly by the latent dynamics.
* **Interpretability.** The latent space is physical, with soliton amplitudes and positions. The end-to-end
  stage keeps a scattering-loss term so the latent stays physical instead of drifting into an arbitrary code.

## Architecture

**Slots.** A variable number of solitons is represented as N fixed slots (κ, x₀, presence), sorted by κ.
Positions use `x₀` rather than `log c`: it is the soliton center when the soliton is isolated, O(domain) in
size, and evolves by a plain translation.

**Encoder `E_θ`.** A strided residual 1-D CNN over `[u, x/L]`. The coordinate channel is needed because
positions are absolute. An MLP head produces presence logits, `κ = softplus(·) + 0.05`, `x₀ = L·(·)`, and
optionally `R` on a fixed k-grid.

**Decoder `D_φ`.**

1. *Renderer.* Each slot is drawn as `fₙ(x) = κₙ² g_φ(κₙ, κₙ(x − x₀ₙ))` by one shared MLP `g_φ`.
   This is translation-equivariant and has KdV's scaling symmetry `u → κ²U(κx)` built in. The *shape* is
   learned; `g_φ` discovers `−2 sech²`.
2. *Interaction network.* A dilated residual CNN with no coordinate channel and a per-position channel norm.
   GroupNorm/BatchNorm statistics over x would break equivariance; a unit test catches exactly that.
   Its inputs are the individual slot fields, their sum and, with radiation, the exact linear Born feature
   `F_c(2x) = (1/2π)∫R(k)e^{2ikx}dk`. It outputs a correction to the summed field. During collisions the
   correction is O(1): the two-soliton profile is not the sum of two solitons.

**Baseline.** `FNO1dBaseline`: lift `[u₀, x/L, t/T]`, then 4 spectral layers (64 modes, width 64), then
project. It is trained on the same data distribution and with the same total number of gradient steps.

## Training (`neural.train`)

1. **Encoder:** scattering loss (presence BCE + masked MSE on κ and x₀) against exact data.
2. **Decoder:** teacher-forced on exact slots evolved to random `t ∈ [0, T]`. The loss is relative L2 against
   the exact `u(x,t)`, computed on the fly from the closed form.
3. **End-to-end:** learned encoder → exact flow → learned decoder. The loss is relative L2 on `u(x,t)`
   + 0.1 × scattering loss (keeps the latent physical) + an optional physics-informed term:
   the KdV residual `u_t − 6uu_x + u_xxx`, with `u_t` by finite differences *through the exact flow* and
   x-derivatives by 4th-order stencils, normalized by `‖u_xxx‖²`.

Data (`generate_kdv_soliton_data`): scattering data are sampled directly, and `u₀` comes from the exact
inverse transform, so every training pair is exact by construction. Samples are rejection-sampled so the
profile stays inside the domain at every checked time. Interaction phase shifts can move a soliton several
widths from its nominal `x₀`, and a truncated profile no longer has the sampled scattering data.
`generate_kdv_general_data` labels arbitrary potentials, including radiation, with the exact direct transform.

## Results

See `examples/neural_ist.py`. Its output (`examples/output/neural_ist_results.json`) is summarized in the
README.

## Limitations and next steps

* The demo trains on reflectionless (soliton) data. The radiation path (R head + Born channel) is
  implemented and tested for shapes and gradients, but it is not benchmarked, because exact Marchenko targets
  cost about 2n⁴ per sample.
* The slot count N caps the representable soliton number. Permutation ambiguity is handled by sorting on κ,
  which becomes ill-conditioned when two κ nearly coincide.
* NLS is not yet wired in. The same construction applies with slots `(ξ, η, x₀, φ)` and the flow
  `x₀ − 4ξt`, `φ + 4(η² − ξ²)t`-type phases.
