"""Korteweg-de Vries equation  u_t - 6 u u_x + u_xxx = 0  via IST.

Pipeline:   u(x,0) --direct--> {kappa_n, c_n, R(k)}
                   --evolve--> {kappa_n, c_n e^{4 kappa_n^3 t}, R(k) e^{8 i k^3 t}}
                   --inverse-> u(x,t)

With this sign convention solitons are wells, u = -2 kappa^2 sech^2(kappa(x - 4 kappa^2 t - x0)),
moving right with speed 4 kappa^2; radiation disperses to the left.

A periodic pseudo-spectral solver (integrating-factor RK4) is included as an
independent reference for validating the IST solution.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .direct import KdVScatteringData, kdv_direct
from .inverse import kdv_inverse, kdv_reflectionless
from .direct import schrodinger_ab
from .utils import Grid, complex_dtype, sech, time_aware_spectral_grid


def _refl(u0: Tensor, k: Tensor, grid: Grid) -> Tensor:
    a, b = schrodinger_ab(u0, k.to(complex_dtype(u0.dtype)), grid)
    return b / a


def kdv_soliton(x: Tensor, t, kappa: float, x0: float = 0.0) -> Tensor:
    """Exact one-soliton u = -2 kappa^2 sech^2(kappa (x - 4 kappa^2 t - x0)).
    t may be a scalar or a 1-D tensor (returns (Nt, Nx))."""
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    xx = x - 4 * kappa ** 2 * t[..., None] - x0 if t.ndim else x - 4 * kappa ** 2 * t - x0
    return -2 * kappa ** 2 * sech(kappa * xx) ** 2


def kdv_two_soliton_phase_shifts(k1: float, k2: float) -> tuple[float, float]:
    """Asymptotic position shifts after a collision (k1 > k2): the faster soliton
    jumps forward by +log((k1+k2)/(k1-k2))/k1 and the slower one backward by
    -log((k1+k2)/(k1-k2))/k2."""
    L = math.log((k1 + k2) / (k1 - k2))
    return L / k1, -L / k2


def kdv_ist_solve(u0: Tensor, x: Tensor, t: Tensor, *, reflection: bool = True,
                  x_out: Tensor | None = None, k: Tensor | None = None,
                  method: str = "auto") -> tuple[Tensor, KdVScatteringData]:
    """Solve KdV by inverse scattering.

    u0: (..., n) initial data on uniform x; t: (Nt,) output times.
    reflection: include the continuous spectrum (False = keep only solitons,
      i.e. the long-time soliton content of u0).
    x_out: reconstruction grid (default x).  Must be uniform for Marchenko.
    k: spectral grid for R(k); by default one fine enough that the evolved
      radiation (group speed 12 k^2) does not alias over the requested times.
    Returns (u, data) with u of shape (..., Nt, Nx_out).
    """
    x_out = x if x_out is None else x_out
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(-1)
    if reflection and k is None:
        grid = Grid.from_x(x)
        # radiation at wavenumber k moves left with speed 12 k^2; s = 2x
        k = time_aware_spectral_grid(grid, lambda kk: _refl(u0, kk, grid), float(t.abs().max()),
                                     lambda kk: 12 * kk ** 2, 2.0, device=x.device, dtype=x.dtype)
    data = kdv_direct(u0, x, k, reflection=reflection)
    return kdv_evolve_and_invert(data, x_out, t, method=method), data


def kdv_evolve_and_invert(data: KdVScatteringData, x: Tensor, t: Tensor, method: str = "auto") -> Tensor:
    """u(x, t) for every t from scattering data at time data.t.  (..., Nt, Nx)."""
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(-1)
    if data.R is None or method == "closed_form":
        return kdv_reflectionless(data.reflectionless().evolve(t), x)
    # Marchenko: one time at a time to bound memory
    return torch.stack([kdv_inverse(data.evolve(float(ti)), x, method=method) for ti in t], -2)


def kdv_from_solitons(x: Tensor, t: Tensor, kappa, x0) -> Tensor:
    """N-soliton solution with amplitudes 2 kappa_n^2 and positions x0 (each the
    position the soliton would have at t = 0 if it were alone).  Differentiable
    in kappa and x0.  Returns (..., Nt, Nx)."""
    data = KdVScatteringData.solitons(kappa, x0, dtype=x.dtype, device=x.device)
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(-1)
    return kdv_reflectionless(data.evolve(t), x)


# ---------------------------------------------------------------------------
# Reference PDE solver
# ---------------------------------------------------------------------------

def kdv_spectral_solve(u0: Tensor, x: Tensor, t: Tensor, dt: float = 1e-3) -> Tensor:
    """Periodic Fourier pseudo-spectral KdV solver (integrating-factor RK4,
    2/3 dealiasing).  u_t = -u_xxx + 3 (u^2)_x.

    u0: (..., n) on uniform x (treated as periodic with period n*h);
    t: (Nt,) increasing output times >= 0.  Returns (..., Nt, n)."""
    grid = Grid.from_x(x)
    k = grid.k(device=x.device, dtype=x.dtype)
    dealias = (k.abs() < (2.0 / 3.0) * k.abs().max()).to(x.dtype)
    L = 1j * k ** 3

    def N(v_hat: Tensor, tau: float) -> Tensor:
        u_hat = v_hat * torch.exp(L * tau)
        u = torch.fft.ifft(u_hat * dealias).real
        return torch.exp(-L * tau) * (3j * k) * torch.fft.fft(u * u) * dealias

    t = torch.as_tensor(t, dtype=x.dtype).reshape(-1).tolist()
    u_hat0 = torch.fft.fft(u0.to(x.dtype))
    v = u_hat0.clone()
    tau = 0.0
    out = []
    for t_target in t:
        while tau < t_target - 1e-12:
            h = min(dt, t_target - tau)
            k1 = N(v, tau)
            k2 = N(v + 0.5 * h * k1, tau + 0.5 * h)
            k3 = N(v + 0.5 * h * k2, tau + 0.5 * h)
            k4 = N(v + h * k3, tau + h)
            v = v + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            tau += h
        out.append(torch.fft.ifft(v * torch.exp(L * tau)).real)
    return torch.stack(out, -2)
