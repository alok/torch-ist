# Derivations and design notes

This file covers the non-obvious mathematics behind `torch-ist`, in the order the code uses it.
Each section ends with the bug or failure mode that the design avoids, because those are the reasons
the code looks the way it does.

## 1. Transfer matrices

Both spectral problems are written as `v' = A(x; λ) v` with `A² = μ² I`.

* Schrödinger, with `v = (ψ, ψ')`: `A = [[0, 1], [u − k², 0]]` and `μ² = u − k²`.
* Zakharov–Shabat (focusing): `A = [[−iζ, q], [−q̄, iζ]]` and `μ² = −ζ² − |q|²`.

Because `A² = μ²I`, `exp(Ah) = cosh(μh) I + (sinh(μh)/μ) A`. Both coefficients are entire functions of `μ²`.
`utils.cosh_sinhc` evaluates them, using a Taylor branch near 0 so that autograd never sees `0/0`.
With the potential constant on each cell, the propagator is exact for that piecewise-constant potential.
This is the exponential midpoint rule: it is symmetric, so its global error expands in `h², h⁴, …`.
`(4 f_h − f_{2h})/3`, where the `2h` grid uses the odd-indexed samples, is therefore O(h⁴) (`direct._richardson`).
The extrapolation only holds in the asymptotic regime `k·2h ≲ 1`. That is why the default spectral cutoff is
`0.8/h` rather than Nyquist. Past that point the extrapolation produced spurious `|r| ≈ 10⁻²` for an exact soliton.

The product `M_{n−1} ⋯ M_0` is reduced pairwise (`utils.ordered_product`). That takes log₂ n batched matmuls,
so it is GPU friendly and differentiable.

## 2. Jost coefficients

KdV: the left Jost solution satisfies `ψ ~ e^{−ikx}` as x → −∞ and `ψ ~ a e^{−ikx} + b e^{ikx}` as x → +∞.
Then `R = b/a` (the right reflection coefficient), `T = 1/a`, bound states satisfy `a(iκ) = 0`, and

    c_n² = b(iκ_n) / (da/dκ)(iκ_n)          ( = 1/∫ψ_n² with ψ_n ~ e^{−κx} at +∞ ).

Check with `−2 sech² x`: `ψ = e^{−ikx}(ik + tanh x)/(ik − 1)`, so `a = (ik+1)/(ik−1)`, `κ = 1`, `b = 1`,
`da/dκ = 1/2`, and `c² = 2`.

NLS: `φ ~ (1,0)e^{−iζx}` at −∞ and `φ ~ (a e^{−iζx}, b e^{iζx})` at +∞. Then `r = b/a` on ℝ,
`a(ζ_n) = 0` with `Im ζ_n > 0`, and `C_n = b(ζ_n)/a'(ζ_n)`.

**Matching for `b`.** At a bound state the eigenfunction decays at both ends. Integrating from one side
only is unstable past the eigenfunction's peak, because roundoff excites the growing mode. So `b` comes from
the left solution integrated rightwards and the right solution integrated leftwards, matched at a point.
That point has to be the peak of *that* eigenfunction. For two separated solitons, the slow soliton's
eigenfunction is about `e^{−κ₂·12}` inside the fast soliton's well. An earlier version matched at the deepest
point of the potential, and its `log c₂` was off by 3.4. Peaks come from eigenvectors of a coarse
finite-difference (KdV) or Fourier-collocation (NLS) operator.

**Differentiable eigenvalues.** Roots are found without gradients. One Newton step is then taken with
gradients enabled:

    κ = κ* − a(κ*; u) / (∂_κ a)(κ*).detach()

Its value is unchanged to first order, and its gradient is `−∂_u a / ∂_κ a`. That is exactly the
implicit-function theorem.

## 3. Marchenko equations

KdV (right equation), for y > x:

    K(x,y) + F(x+y) + ∫_x^∞ K(x,z) F(z+y) dz = 0,
    F(s) = Σ c_n² e^{−κ_n s} + (1/2π)∫ R(k) e^{iks} dk,        u = −2 d/dx K(x,x).

NLS. Start from `φ/a = ψ̄ + rψ`, multiply by `e^{iξy}/2π`, and integrate. Closing the contour in the
upper half plane picks up the residues at `ζ_n`. This gives

    K̄(x,y) + (0,1)ᵀF(x+y) + ∫_x^∞ K(x,s) F(s+y) ds = 0,
    F(s) = (1/2π)∫ r(ξ)e^{iξs}dξ − i Σ C_n e^{iζ_n s}.

The focusing symmetry `(v₁,v₂)(ζ) ↦ (v̄₂, −v̄₁)(ζ̄)` gives `ψ̄(x,ζ) = (ψ̄₂(ζ̄), −ψ̄₁(ζ̄))`, so `K̄ = (K̄₂, −K̄₁)`.
Writing `G = K̄₁` (the conjugate of K₁) and `L = −∫ G F̄`:

    G = F + L H,   L = −G H̄          ⇔   [G L] [[I, H̄], [−H, I]] = [f, 0],
    q(x) = −2 conj(G(x,x)).

Here `H(s,y) = F(s+y)` is a complex-symmetric Hankel operator, so `I + H̄H = I + HᴴH ≥ I`.
A wrong sign choice at this step yields `1 − |C|²e^{−4ηx}/4η²` in the one-soliton denominator, which is singular.
The correct sign gives `1 + …` and `q = 2η sech(2η(x−x₀))`.

## 4. The scaled low-rank (Woodbury) solve

Discretize on `z ∈ [x, Z]` with symmetric weights `W^{1/2}`. The soliton part of the kernel factorizes:

    KdV:  F_d(z+y) = Σ λ_n e^{−κ_n(z−x)} e^{−κ_n(y−x)},   λ_n = c_n² e^{−2κ_n x},
    NLS:  F_d(z+y) = Σ λ_n e^{iζ_n(z−x)} e^{iζ_n(y−x)},   λ_n = −iC_n e^{2iζ_n x}.

So the system becomes `X (M_c + 𝒰 𝒟 𝒱ᵀ) = f_c + ρ 𝒟 𝒱ᵀ`, where `M_c` holds only the continuous spectrum.
For KdV, `M_c = I + F_c`, `𝒰 = 𝒱 = Ũ`, `𝒟 = Λ`, and `ρ = −1ᵀ`. For NLS, `M_c = [[I, H̄_c], [−H_c, I]]`,
`𝒰 = diag(Ū, Ũ)`, `𝒱 = diag(Ũ, Ū)`, `𝒟 = [[0, Λ̄], [−Λ, 0]]`, and `ρ = [0, −1ᵀ]`.
Woodbury in push-through form gives

    X e₀ = β + (ρ − a) Y γ,     Y = 𝒟 (I + P𝒟)⁻¹,
    β = f_c M_c⁻¹ e₀,  a = f_c M_c⁻¹ 𝒰,  P = 𝒱ᵀ M_c⁻¹ 𝒰,  γ = 𝒱ᵀ M_c⁻¹ e₀.

The two `ρ𝒟γ` terms cancel exactly.

`|λ_n| ~ e^{2κ|x − x_n|}` overflows and wrecks conditioning. The fix is to factor `𝒟 = 𝒮 𝒜₁ 𝒜₂ 𝒮`,
where `𝒮 = diag(σ^{1/2})` with `σ = max(|λ|, 1)`, and `𝒜₁`, `𝒜₂` are bounded and commute with `𝒮`. Then

    Y = 𝒜₁ (𝒮⁻² + 𝒜₂ P 𝒜₁)⁻¹ 𝒜₂.

Every entry of that matrix is O(1). For KdV, `𝒜₁ = I` and `𝒜₂ = Λ/σ`. For NLS, `𝒜₁ = J = [[0, I], [−I, 0]]`
and `𝒜₂ = diag(Φ, Φ̄)` with `Φ = Λ/σ`. The identity holds for every choice of σ, so the max kink does not
disturb autograd.

With `F_c = 0`, the identity part of `P` is the exact Gram matrix: `1/(κ_m+κ_n)` for KdV and
`i/(ζ_m − ζ̄_n)` for NLS. This recovers the closed forms in `inverse.kdv_reflectionless` and
`inverse.nls_reflectionless`. Since `K(x,x) = ∂ₓ log det(I + A)`, the KdV closed form is
`u = −2∂²ₓ log det` computed without forming the determinant. In the general solver, `P` is assembled as
(exact Gram) + `𝒱ᵀ(M_c⁻¹ − I)𝒰`, so only the radiation coupling is quadrature.

The NLS Gram sign gets its own test, `test_roundtrip_nls_marchenko`. With no radiation, `P` enters `Y` only
through products `Γ Λ̄ Γ̄ Λ`, so a sign error in Γ is invisible in every pure-soliton test. It surfaced only
when solitons and radiation were present together.

## 5. Quadrature windows

`F(z+y)` is O(1) wherever `z + y` lies left of the potential. The z-integral therefore cannot stop at the grid
edge. It must run to `z = 2x_max − x`, beyond which `F(z+y) = 0` for every `y ≥ x`. Truncating at `x_max`
gave O(1) errors near the left edge for repulsive potentials. Each `x_i` gets a local window
`z_l = x_i + lh`, `l ≤ 2(n−1−i)`. On that window the kernel is a shifted Hankel gather `F_c[2i + l₁ + l₂]`.
The windows use Gregory end-corrected trapezoid weights `h·[3/8, 7/6, 23/24, 1, …]`.
`u = −2∂ₓK` uses 4th-order central differences.

## 6. Spectral grids

* For a weak KdV potential, R(k) goes from −1 at k = 0 to about 0 over the scale `k ~ ∫u`. A uniform grid misses
  this, and F(s) acquires a spurious tail. `graded_spectral_grid` uses geometric spacing from `k = 10⁻³`.
* The quadrature makes `F(s)` periodic with period `2π/dk`. Over the needed `s`-range of length `8L` that requires
  `dk ≤ π/6L`. After evolution, radiation at wavenumber k lives near `s = 2x ≈ −24k²t` for KdV (`8|ξ|t` for NLS).
  `time_aware_spectral_grid` shrinks `dk` to cover this. It sizes the grid from the largest `k` where `|R|` is
  non-negligible, rather than from `k_max`.

## 7. Time evolution

The `t`-part of the Lax pair at `|x| → ∞` makes the Jost data evolve as follows.

* KdV `u_t − 6uu_x + u_xxx = 0`: `a` is fixed, `b → b e^{8ik³t}`, so `R → R e^{8ik³t}` and `c_n² → c_n² e^{8κ_n³t}`.
  Check: `c² = 2κe^{2κx₀}` makes the center move at `4κ²`.
* NLS: `b → b e^{4iζ²t}`, so `C_n → C_n e^{4iζ_n²t}` and `r → r e^{4iξ²t}`. Check: `ζ = i/2` gives a phase of
  `e^{+it}` for `sech x`, and `ξ ≠ 0` gives velocity `−4ξ`. Both signs are verified against exact solitons
  and against the split-step solver in the tests.

## 8. Phase shifts

With the right-Marchenko normalization, the **rightmost** soliton sits at its free position. So for KdV at
t → +∞ the fast soliton is unshifted and the slow one lags by `log((κ₁+κ₂)/(κ₁−κ₂))/κ₂`. At t → −∞ the
roles swap. The net collision shifts are `+L/κ₁` and `−L/κ₂`, and `two_soliton_collision.py` measures them
to 4 digits.
