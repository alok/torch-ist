"""Focusing nonlinear Schrodinger equation  i q_t + q_xx + 2 |q|^2 q = 0  via IST.

Lax pair x-part: Zakharov-Shabat  v_x = [[-i zeta, q], [-conj(q), i zeta]] v.

Pipeline:   q(x,0) --direct--> {zeta_n, C_n, r(xi)}
                   --evolve--> {zeta_n, C_n e^{4 i zeta_n^2 t}, r(xi) e^{4 i xi^2 t}}
                   --inverse-> q(x,t)

One soliton (zeta = xi + i eta):
    q = 2 eta sech(2 eta (x - x0 + 4 xi t)) exp(-2 i xi x - 4 i (xi^2 - eta^2) t + i phi),
so u(x,0) = sech(x) is the eta = 1/2, xi = 0 standing soliton sech(x) e^{it}.

A split-step Fourier (Strang) solver is included as an independent reference.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .direct import NLSScatteringData, nls_direct
from .inverse import nls_inverse, nls_reflectionless
from .utils import Grid, complex_dtype, sech


def nls_soliton(x: Tensor, t, eta: float = 0.5, xi: float = 0.0, x0: float = 0.0, phase: float = 0.0) -> Tensor:
    """Exact bright soliton (see module docstring).  t scalar or (Nt,) -> (Nt, Nx)."""
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    tt = t[..., None] if t.ndim else t
    arg = 2 * eta * (x - x0 + 4 * xi * tt)
    ph = -2 * xi * x - 4 * (xi ** 2 - eta ** 2) * tt + phase
    return (2 * eta * sech(arg)) * torch.exp(1j * ph.to(complex_dtype(x.dtype)))


def nls_ist_solve(q0: Tensor, x: Tensor, t: Tensor, *, reflection: bool = True,
                  x_out: Tensor | None = None, xi: Tensor | None = None,
                  method: str = "auto") -> tuple[Tensor, NLSScatteringData]:
    """Solve the focusing NLS by inverse scattering.  q0: (..., n) on uniform x.
    Returns (q, data) with q of shape (..., Nt, Nx_out) complex."""
    x_out = x if x_out is None else x_out
    data = nls_direct(q0, x, xi, reflection=reflection)
    return nls_evolve_and_invert(data, x_out, t, method=method), data


def nls_evolve_and_invert(data: NLSScatteringData, x: Tensor, t: Tensor, method: str = "auto") -> Tensor:
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(-1)
    if data.r is None or method == "closed_form":
        return nls_reflectionless(data.reflectionless().evolve(t), x)
    return torch.stack([nls_inverse(data.evolve(float(ti)), x, method=method) for ti in t], -2)


def nls_from_solitons(x: Tensor, t: Tensor, eta, xi=0.0, x0=0.0, phase=0.0) -> Tensor:
    """Multi-soliton (reflectionless) solution, differentiable in all parameters.
    For a single soliton this is exactly `nls_soliton`.  Returns (..., Nt, Nx)."""
    data = NLSScatteringData.solitons(eta, xi, x0, phase, dtype=complex_dtype(x.dtype), device=x.device)
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(-1)
    return nls_reflectionless(data.evolve(t), x)


# ---------------------------------------------------------------------------
# Reference PDE solver
# ---------------------------------------------------------------------------

def nls_split_step_solve(q0: Tensor, x: Tensor, t: Tensor, dt: float = 1e-3) -> Tensor:
    """Periodic split-step Fourier (Strang, 2nd order) solver.
    q0: (..., n); t: (Nt,) increasing >= 0.  Returns (..., Nt, n) complex."""
    grid = Grid.from_x(x)
    k = grid.k(device=x.device, dtype=x.dtype)
    q = q0.to(complex_dtype(x.dtype))
    times = torch.as_tensor(t, dtype=x.dtype).reshape(-1).tolist()
    tau = 0.0
    out = []

    def lin(q: Tensor, h: float) -> Tensor:
        return torch.fft.ifft(torch.fft.fft(q) * torch.exp(-1j * k ** 2 * h))

    for t_target in times:
        while tau < t_target - 1e-12:
            h = min(dt, t_target - tau)
            q = lin(q, 0.5 * h)
            q = q * torch.exp(2j * (q.abs() ** 2) * h)
            q = lin(q, 0.5 * h)
            tau += h
        out.append(q)
    return torch.stack(out, -2)
