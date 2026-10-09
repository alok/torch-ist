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

**Encoder.** Two options:

* `SolitonDetector` (default). Fully convolutional, CenterNet-style: at every grid point it predicts a
  soliton logit, κ, and a sub-grid offset, and slots are the top local maxima. It is translation-equivariant
  (no coordinate channel; per-position norm), works for any soliton count up to N, and is trained with a focal
  heatmap loss plus regression.
* `ScatteringEncoder`. A strided CNN over `[u, x/L]` followed by an MLP with slot heads: presence logits,
  `κ = softplus(·)+0.05`, `x₀ = L·(·)`, and optionally `R` on a fixed k-grid.

**Decoder `D_φ`.**

1. *Pairwise phase shifts in scattering space.* In right-Marchenko coordinates, soliton n sits at
   `x₀ₙ − (1/κₙ)Σ_{m right of n} Φ(κₘ/κₙ)`, even when the solitons are far apart. `Φ_θ` is an MLP of the ratio
   (features r and log|1−r|), applied with a learned soft ordering. It is permutation-equivariant,
   count-agnostic, and scale-invariant.
2. *Renderer.* `fₙ(x) = κₙ² g_φ(ξ) sech ξ` with `ξ = κₙ(x − xₙ)`: one universal profile, built on KdV's
   scaling symmetry, inside an exponential-localization envelope. It is translation-equivariant and
   extrapolates in amplitude.
3. *Collision network.* A dilated CNN with **no biases and no normalization**, so it is continuous at 0 with
   value 0. A scale-invariant norm would amplify a 1e-6 input back up to O(1); a unit test caught exactly that.
   It is fed only `Σ_{i<j}fᵢfⱼ` and `Σ_{i≠j}fᵢ²fⱼ`, which vanish unless solitons overlap pointwise, plus the
   exact Born feature `F_c(2x)` when radiation is modeled. Its output is therefore exactly zero for separated
   solitons, so it only reshapes collision cores. There is no coordinate channel.

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

`examples/neural_ist.py`, run on Apple MPS in float32. Training data: 6000 random 1–3-soliton KdV initial
conditions, κ ∈ [0.4, 1.2], t ∈ [0, 2], on a 384-point grid over [−20, 30]. The neural IST takes 4000 encoder
steps, 4000 decoder steps, and 2500 end-to-end steps. The FNO takes 10 500 steps, which is the same total
number of gradient steps. Metric: mean relative L2 error of u(x,t) on 256 held-out initial conditions per set.

| test set | exact/exact | exact/learned | learned/exact | learned/learned | FNO |
|---|---|---|---|---|---|
| in distribution, t ≤ 2 | 3e-5 | **1.3e-2** | 5.6e-2 | 4.9e-2 | **1.1e-2** |
| time extrapolation, 2 < t ≤ 4 | 7e-5 | **4.6e-3** | 4.6e-2 | **4.5e-2** | 1.49 |
| 4 solitons (trained on ≤ 3) | 4e-5 | **0.14** | 0.64 | 0.63 | **0.13** |
| κ ∈ [1.2, 1.5] (trained on ≤ 1.2) | 4e-4 | **5.1e-2** | 0.52 | 0.49 | 0.43 |

Ablation: the encoder from the previous iteration (CNN + MLP slot heads; `--encoder slots`) gives
learned/learned = 2.2e-2, 3.1e-2, 0.70, 0.66, and gets 100% of in-distribution soliton counts right. The
detector has 7× better position accuracy (median |Δx₀| 2.2e-3 vs 1.6e-2) but 95% count accuracy. It also
misses the 4th soliton in 89% of the 4-soliton set: those failures concentrate where two x₀ nearly coincide,
which produces one fused bump.

**What the decoder learned.** The pairwise phase-shift network recovers the exact KdV law
`Φ(r) = log((1+r)/|1−r|)` to 0.08 (max error over the training range, |1−r| > 0.1). The universal profile
`g_θ(ξ)sech ξ` recovers `−2sech²ξ` to 3e-4. Neither was supplied; both came out of fitting u(x,t).
See `examples/output/neural_ist_learned_laws.png`.

**Takeaways.**

1. **Time extrapolation is the headline.** Moving from t ≤ 2 to t ≤ 4, FNO goes from 1.1e-2 to 1.49 (it falls
   apart). The scattering-space operator stays flat or improves (exact/learned 1.3e-2 → 4.6e-3, because
   separated solitons are easier than colliding ones). Time never enters a network.
2. **The learned inverse transform generalizes** to unseen soliton counts (0.14, on par with FNO) and to
   unseen amplitudes (5.1e-2, versus FNO 0.43). This follows from the symmetries built into it: a universal
   profile from the scaling symmetry, permutation-invariant pairwise phase shifts, and a collision network
   that is exactly zero for separated solitons.
3. **The learned direct transform is the bottleneck.** In distribution, FNO is about 2–4× more accurate than
   the fully learned pipeline. Out of distribution in amplitude or count, the encoder fails, because a CNN
   never shown κ > 1.2 or four solitons does not invent them. When it fails, though, it fails *physically*:
   it outputs clean solitons with the wrong κ, while FNO outputs noise.
4. **Speed.** Out of the box, the exact direct transform (bound-state search, run per sample in float64 on
   CPU) dominates exact-IST cost. Replacing it with the learned encoder gives a 12× speedup at about 5% error.
   For pure solitons the exact *inverse* is a cheap closed form, so a learned inverse only pays off for
   radiation, where Marchenko costs about 2n⁴ per sample.

### Iteration history (what failed and why)

| version | change | 4-soliton (exact/learned) | κ out of range (exact/learned) |
|---|---|---|---|
| v1 | per-slot channels, renderer sees κ | 0.48 | 0.30 |
| v2 | permutation-invariant channels, κ-free renderer | 0.41 | 0.35 |
| v3 | learned pairwise phase shifts, localized renderer, bias-free collision CNN | **0.14** | **0.057** |

v2 barely helped. The real problem was that in right-Marchenko coordinates a soliton's position depends on
the solitons to its right *even when they are far apart*. Only a mechanism that acts in scattering space
(the pairwise Φ) can express that. A CNN can only express it within its receptive field.

## Limitations and next steps

* The demo trains on reflectionless (soliton) data. The radiation path (R head + Born channel) is
  implemented and tested for shapes and gradients, but it is not benchmarked, because exact Marchenko targets
  cost about 2n⁴ per sample.
* The slot count N caps the representable soliton number.
* The learned encoder does not extrapolate in amplitude or in soliton count. A promising fix is to predict
  *visible* positions (which are local) and convert them to x₀ with the decoder's learned Φ, since x₀ is
  non-local in right-Marchenko coordinates. Scale-equivariant convolutions could address amplitude.
* NLS is not yet wired in. The same construction applies with slots `(ξ, η, x₀, φ)` and the flow
  `x₀ − 4ξt`, `φ + 4(η² − ξ²)t`-type phases.
