"""Grids, dtype helpers and the batched 2x2 transfer-matrix machinery.

Both scattering problems in this package are first-order 2x2 linear ODEs

    v' = A(x; lam) v,        A(x; lam)^2 = mu2(x; lam) * I,

(Schrodinger written for (psi, psi') and Zakharov-Shabat).  With the
potential held constant on each grid cell, the cell propagator is exact:

    exp(A h) = cosh(mu h) I + sinh(mu h)/mu * A,       mu = sqrt(mu2).

The total transfer matrix is the ordered product of the cell propagators,
which we reduce pairwise in O(log N) batched matmuls so it vectorises over
spectral parameters, batch elements and devices, and stays differentiable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

DEFAULT_DTYPE = torch.float64


def complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """Complex dtype matching a real dtype (float32 -> complex64, ...)."""
    if dtype.is_complex:
        return dtype
    return {torch.float32: torch.complex64, torch.float64: torch.complex128}[dtype]


def real_dtype(dtype: torch.dtype) -> torch.dtype:
    if not dtype.is_complex:
        return dtype
    return {torch.complex64: torch.float32, torch.complex128: torch.float64}[dtype]


@dataclass(frozen=True)
class Grid:
    """Uniform grid x_j = x_min + j*h, j = 0..n-1.

    The potential sample u_j is taken constant on the cell
    [x_j - h/2, x_j + h/2], so the scattering domain is [x_left, x_right].
    """

    x_min: float
    x_max: float
    n: int

    @property
    def h(self) -> float:
        return (self.x_max - self.x_min) / (self.n - 1)

    @property
    def x_left(self) -> float:
        return self.x_min - 0.5 * self.h

    @property
    def x_right(self) -> float:
        return self.x_max + 0.5 * self.h

    @property
    def length(self) -> float:
        return self.x_right - self.x_left

    def x(self, device: torch.device | str | None = None, dtype: torch.dtype = DEFAULT_DTYPE) -> Tensor:
        return torch.linspace(self.x_min, self.x_max, self.n, device=device, dtype=dtype)

    def k(self, device: torch.device | str | None = None, dtype: torch.dtype = DEFAULT_DTYPE) -> Tensor:
        """Angular wavenumbers of the periodic FFT grid (period n*h)."""
        return 2 * math.pi * torch.fft.fftfreq(self.n, d=self.h, device=device, dtype=dtype)

    @staticmethod
    def from_x(x: Tensor) -> "Grid":
        if x.ndim != 1 or x.numel() < 3:
            raise ValueError("x must be a 1-D grid with at least 3 points")
        h = (x[-1] - x[0]).item() / (x.numel() - 1)
        dx = torch.diff(x)
        if (dx - h).abs().max().item() > 1e-8 * max(1.0, abs(h)):
            raise ValueError("x must be uniformly spaced")
        return Grid(float(x[0]), float(x[-1]), x.numel())


def spectral_grid(k_max: float, n: int, device=None, dtype: torch.dtype = DEFAULT_DTYPE) -> Tensor:
    """Symmetric real spectral grid on (-k_max, k_max) that avoids k = 0.

    Midpoints of n equal cells, so trapezoid == midpoint rule and
    the grid is closed under k -> -k.
    """
    dk = 2 * k_max / n
    return -k_max + dk * (torch.arange(n, device=device, dtype=dtype) + 0.5)


def graded_spectral_grid(k_max: float, dk: float, k_min: float = 1e-3, rel: float = 0.02,
                         device=None, dtype: torch.dtype = DEFAULT_DTYPE) -> Tensor:
    """Symmetric real grid excluding 0: geometric (relative spacing `rel`)
    from k_min up to where the spacing reaches dk, then uniform with step dk.

    The reflection coefficient of a weak KdV potential jumps from R(0) = -1 to
    ~0 over a k-scale ~ int u dx, which a uniform grid cannot resolve; the
    graded part captures it, the uniform part bounds aliasing of F(s)."""
    k1 = dk / rel
    pos: list[float] = []
    k = k_min
    while k < min(k1, k_max):
        pos.append(k)
        k *= 1 + rel
    k = pos[-1] + dk if pos else dk
    while k <= k_max:
        pos.append(k)
        k += dk
    kp = torch.tensor(pos, dtype=dtype, device=device)
    return torch.cat([-kp.flip(0), kp])


def default_spectral_grid(grid: Grid, k_max: float | None = None, dk: float | None = None,
                          device=None, dtype: torch.dtype = DEFAULT_DTYPE) -> Tensor:
    """Graded spectral grid whose coarse spacing dk = pi/(6 max|x|) keeps the
    quadrature for F(s) alias-free on the range of s = z + y the Marchenko
    solver needs (length 8 max|x|)."""
    L = max(abs(grid.x_left), abs(grid.x_right))
    if k_max is None:
        # Richardson uses the 2h grid too; keep k * 2h < ~1.6 so both grids
        # are in the asymptotic regime (beyond this the extrapolation is noise)
        k_max = min(0.8 / grid.h, 25.0)
    if dk is None:
        dk = math.pi / (6.0 * L)
    return graded_spectral_grid(k_max, dk, device=device, dtype=dtype)


def quadrature_weights(k: Tensor) -> Tensor:
    """Trapezoid weights for a sorted, possibly non-uniform grid."""
    w = torch.zeros_like(k)
    d = torch.diff(k)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return w


def trapezoid_weights(n: int, h: float, device=None, dtype=DEFAULT_DTYPE) -> Tensor:
    w = torch.full((n,), h, device=device, dtype=dtype)
    w[0] = w[-1] = 0.5 * h
    return w


# ---------------------------------------------------------------------------
# 2x2 propagators
# ---------------------------------------------------------------------------

def cosh_sinhc(mu2: Tensor, h: float) -> tuple[Tensor, Tensor]:
    """(cosh(mu h), sinh(mu h)/mu) as entire functions of mu2 = mu^2.

    Uses a Taylor series near mu2 = 0, with the far branch fed a safe value so
    autograd never sees 0/0 (the usual torch.where NaN trap).
    """
    z = mu2 * (h * h)
    small = z.abs() < 1e-6
    z_safe = torch.where(small, torch.ones_like(z), z)
    mu_h = torch.sqrt(z_safe)
    c_big = torch.cosh(mu_h)
    s_big = torch.sinh(mu_h) / mu_h * h
    c_small = 1 + z / 2 + z * z / 24
    s_small = h * (1 + z / 6 + z * z / 120)
    return torch.where(small, c_small, c_big), torch.where(small, s_small, s_big)


def cell_propagators(A: Tensor, mu2: Tensor, h: float) -> Tensor:
    """exp(A h) for a stack of 2x2 matrices with A^2 = mu2 I.

    A: (..., 2, 2) complex, mu2: (...) complex.
    """
    c, s = cosh_sinhc(mu2, h)
    eye = torch.eye(2, dtype=A.dtype, device=A.device)
    return c[..., None, None] * eye + s[..., None, None] * A


def ordered_product(M: Tensor) -> Tensor:
    """M[..., n-1, :, :] @ ... @ M[..., 0, :, :] by pairwise tree reduction.

    O(log n) batched matmuls, so it is GPU friendly and differentiable.
    """
    n = M.shape[-3]
    while n > 1:
        if n % 2:
            eye = torch.eye(2, dtype=M.dtype, device=M.device).expand(*M.shape[:-3], 1, 2, 2)
            M = torch.cat([M, eye], dim=-3)
            n += 1
        M = M[..., 1::2, :, :] @ M[..., 0::2, :, :]
        n //= 2
    return M[..., 0, :, :]


def chunked(n: int, chunk: int):
    for i in range(0, n, chunk):
        yield slice(i, min(i + chunk, n))


def sech(x: Tensor) -> Tensor:
    return 1.0 / torch.cosh(x)


def as_tensor(u, dtype: torch.dtype | None = None, device=None) -> Tensor:
    if isinstance(u, Tensor):
        return u.to(device=device if device is not None else u.device,
                    dtype=dtype if dtype is not None else u.dtype)
    return torch.as_tensor(u, dtype=dtype or DEFAULT_DTYPE, device=device)
