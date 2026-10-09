"""Direct scattering transforms.

KdV  (u_t - 6 u u_x + u_xxx = 0)  ->  Schrodinger operator
    -psi'' + u psi = k^2 psi.
  Left Jost solution psi ~ e^{-ikx} (x -> -inf) and, as x -> +inf,
    psi ~ a(k) e^{-ikx} + b(k) e^{ikx}.
  Scattering data: R(k) = b/a (right reflection coefficient), bound states
  k = i kappa_n with a(i kappa_n) = 0, and right norming constants
    c_n^2 = b(i kappa_n) / (da/dkappa)(i kappa_n),
  equivalently psi_n ~ c_n e^{-kappa_n x} at +inf with ||psi_n||_2 = 1.

Focusing NLS (i q_t + q_xx + 2|q|^2 q = 0)  ->  Zakharov-Shabat system
    v_x = [[-i zeta, q], [-conj(q), i zeta]] v.
  Jost solution phi ~ (1, 0) e^{-i zeta x} (x -> -inf) and, as x -> +inf,
    phi ~ (a(zeta) e^{-i zeta x}, b(zeta) e^{i zeta x}).
  Scattering data: r(xi) = b/a on the real line, eigenvalues zeta_n in the
  upper half plane with a(zeta_n) = 0, and C_n = b(zeta_n) / a'(zeta_n).

Everything is computed with exact piecewise-constant transfer matrices
(see utils.py), batched over the spectral parameter and any leading batch
dimensions of the potential.  Eigenvalues are located without gradients and
then polished by one differentiable Newton step, which yields the exact
implicit-function-theorem derivative d(eigenvalue)/d(potential).  That makes
the whole map potential -> scattering data differentiable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .utils import (
    Grid,
    cell_propagators,
    chunked,
    complex_dtype,
    default_spectral_grid,
    ordered_product,
)

# max (spectral points x grid cells) per transfer-matrix chunk
_CHUNK_ELEMS = 2_000_000


# ---------------------------------------------------------------------------
# Scattering-data containers
# ---------------------------------------------------------------------------

@dataclass
class KdVScatteringData:
    """Scattering data of the Schrodinger operator -d^2/dx^2 + u.

    kappa : (..., N) bound-state parameters (eigenvalues -kappa^2), padded.
    log_c : (..., N) log of the right norming constants c_n.
    mask  : (..., N) bool, False on padding.
    k, R  : real spectral grid (K,) and reflection coefficient (..., K), or None
            for reflectionless data.
    t     : time the data refers to.
    """

    kappa: Tensor
    log_c: Tensor
    mask: Tensor
    k: Tensor | None = None
    R: Tensor | None = None
    t: float = 0.0

    @property
    def c2(self) -> Tensor:
        return torch.exp(2 * self.log_c)

    @property
    def num_solitons(self) -> Tensor:
        return self.mask.sum(-1)

    def evolve(self, t) -> "KdVScatteringData":
        """KdV flow u_t - 6uu_x + u_xxx = 0 in scattering space (dt = t - self.t):
        kappa fixed, c_n^2 -> c_n^2 e^{8 kappa^3 dt}, R -> R e^{8 i k^3 dt}.

        t may be a float or a 1-D tensor of times, in which case a time axis is
        inserted before the last dimension of every field: (..., Nt, N)."""
        if isinstance(t, Tensor) and t.ndim == 1:
            dt = (t.to(self.kappa.dtype).to(self.kappa.device) - self.t)[:, None]
            kappa = self.kappa[..., None, :].expand(*self.kappa.shape[:-1], dt.shape[0], self.kappa.shape[-1])
            mask = self.mask[..., None, :].expand(kappa.shape)
            log_c = self.log_c[..., None, :] + 4 * kappa ** 3 * dt
            R = None
            if self.R is not None:
                R = self.R[..., None, :] * torch.exp(8j * self.k.to(self.R.dtype) ** 3 * dt.to(self.R.dtype))
            return replace(self, kappa=kappa, log_c=log_c, mask=mask, R=R, t=float("nan"))
        dt = float(t) - self.t
        log_c = self.log_c + 4 * self.kappa ** 3 * dt
        R = None
        if self.R is not None:
            R = self.R * torch.exp(8j * self.k.to(self.R.dtype) ** 3 * dt)
        return replace(self, log_c=log_c, R=R, t=float(t))

    def reflectionless(self) -> "KdVScatteringData":
        return replace(self, k=None, R=None)

    def to(self, device) -> "KdVScatteringData":
        return KdVScatteringData(
            self.kappa.to(device), self.log_c.to(device), self.mask.to(device),
            None if self.k is None else self.k.to(device),
            None if self.R is None else self.R.to(device), self.t)

    @staticmethod
    def solitons(kappa, x0, dtype=torch.float64, device=None) -> "KdVScatteringData":
        """Reflectionless data for solitons -2 kappa^2 sech^2(kappa (x - x0)) at t = 0.

        For a single soliton c^2 = 2 kappa e^{2 kappa x0}; for several, x0 is the
        position each soliton would have if it were alone (asymptotic phase is
        then shifted by the interaction).
        """
        kappa = torch.as_tensor(kappa, dtype=dtype, device=device)
        x0 = torch.as_tensor(x0, dtype=dtype, device=device)
        kappa, x0 = (torch.atleast_1d(v) for v in torch.broadcast_tensors(kappa, x0))
        log_c = 0.5 * torch.log(2 * kappa) + kappa * x0
        return KdVScatteringData(kappa, log_c, torch.ones_like(kappa, dtype=torch.bool))


@dataclass
class NLSScatteringData:
    """Scattering data of the Zakharov-Shabat system (focusing NLS).

    zeta  : (..., N) complex eigenvalues in the upper half plane, padded.
    log_C : (..., N) complex log of C_n = b(zeta_n)/a'(zeta_n).
    mask  : (..., N) bool.
    xi, r : real spectral grid (K,) and reflection coefficient r = b/a (..., K).
    """

    zeta: Tensor
    log_C: Tensor
    mask: Tensor
    xi: Tensor | None = None
    r: Tensor | None = None
    t: float = 0.0

    @property
    def C(self) -> Tensor:
        return torch.exp(self.log_C)

    @property
    def num_solitons(self) -> Tensor:
        return self.mask.sum(-1)

    def evolve(self, t) -> "NLSScatteringData":
        """Focusing NLS flow i q_t + q_xx + 2|q|^2 q = 0 in scattering space:
        zeta fixed, C_n -> C_n e^{4 i zeta_n^2 dt}, r -> r e^{4 i xi^2 dt}.
        t: float or 1-D tensor (adds a time axis, see KdVScatteringData.evolve)."""
        if isinstance(t, Tensor) and t.ndim == 1:
            dt = (t.to(self.zeta.real.dtype).to(self.zeta.device) - self.t)[:, None].to(self.zeta.dtype)
            zeta = self.zeta[..., None, :].expand(*self.zeta.shape[:-1], dt.shape[0], self.zeta.shape[-1])
            mask = self.mask[..., None, :].expand(zeta.shape)
            log_C = self.log_C[..., None, :] + 4j * zeta ** 2 * dt
            r = None
            if self.r is not None:
                r = self.r[..., None, :] * torch.exp(4j * self.xi.to(self.r.dtype) ** 2 * dt)
            return replace(self, zeta=zeta, log_C=log_C, mask=mask, r=r, t=float("nan"))
        dt = float(t) - self.t
        log_C = self.log_C + 4j * self.zeta ** 2 * dt
        r = None
        if self.r is not None:
            r = self.r * torch.exp(4j * self.xi.to(self.r.dtype) ** 2 * dt)
        return replace(self, log_C=log_C, r=r, t=float(t))

    def reflectionless(self) -> "NLSScatteringData":
        return replace(self, xi=None, r=None)

    def to(self, device) -> "NLSScatteringData":
        return NLSScatteringData(
            self.zeta.to(device), self.log_C.to(device), self.mask.to(device),
            None if self.xi is None else self.xi.to(device),
            None if self.r is None else self.r.to(device), self.t)

    @staticmethod
    def solitons(eta, xi=0.0, x0=0.0, phase=0.0, dtype=torch.complex128, device=None) -> "NLSScatteringData":
        """Reflectionless data for bright solitons
            q = 2 eta sech(2 eta (x - x0)) exp(-2 i xi x + i phase)   at t = 0
        (single-soliton parametrisation; zeta = xi + i eta, velocity -4 xi).
        """
        rdt = torch.float64 if dtype == torch.complex128 else torch.float32
        eta, xi, x0, phase = (torch.as_tensor(v, dtype=rdt, device=device) for v in (eta, xi, x0, phase))
        eta, xi, x0, phase = (torch.atleast_1d(v) for v in torch.broadcast_tensors(eta, xi, x0, phase))
        zeta = torch.complex(xi, eta)
        # q = -2 conj(lambda)/(1+...) with lambda = -i C e^{2 i zeta x}:
        # |C| = 2 eta e^{2 eta x0},  arg: q phase = -arg(-iC) - 2 xi x  => arg C = -phase - pi/2
        # plus the shift from e^{2i zeta x} evaluated with x0 (-2 i xi x0 is carried by x).
        log_abs = torch.log(2 * eta) + 2 * eta * x0
        arg = -phase - math.pi / 2
        log_C = torch.complex(log_abs, arg.to(rdt))
        return NLSScatteringData(zeta, log_C, torch.ones_like(eta, dtype=torch.bool))


# ---------------------------------------------------------------------------
# Transfer matrices and Jost coefficients
# ---------------------------------------------------------------------------

def _chunk_size(n_cells: int, batch: int) -> int:
    return max(1, _CHUNK_ELEMS // max(1, n_cells * batch))


def _schrodinger_ab_raw(u: Tensor, k: Tensor, grid: Grid) -> tuple[Tensor, Tensor]:
    """Jost coefficients a(k), b(k) of -psi'' + u psi = k^2 psi.

    u: (..., n) real potential samples on `grid`; k: (K,) complex, Im k >= 0
    (or real).  Returns a, b of shape (..., K).
    """
    cdt = complex_dtype(u.dtype)
    k = k.to(dtype=cdt, device=u.device)
    batch = u.shape[:-1]
    uc = u.to(cdt)
    outs_a, outs_b = [], []
    nb = int(torch.tensor(batch).prod().item()) if batch else 1
    for sl in chunked(k.numel(), _chunk_size(grid.n, nb)):
        kk = k[sl]                                           # (Kc,)
        mu2 = uc[..., None, :] - (kk ** 2)[:, None]          # (..., Kc, n)
        A = torch.zeros(*mu2.shape, 2, 2, dtype=cdt, device=u.device)
        A[..., 0, 1] = 1
        A[..., 1, 0] = mu2
        T = ordered_product(cell_propagators(A, mu2, grid.h))  # (..., Kc, 2, 2)
        ik = 1j * kk
        # start: psi = e^{-ik x_l} (1, -ik); work with hat-psi = psi e^{ik x_l}
        psi = T[..., 0, 0] - ik * T[..., 0, 1]
        dpsi = T[..., 1, 0] - ik * T[..., 1, 1]
        a = torch.exp(ik * grid.length) * (ik * psi - dpsi) / (2 * ik)
        b = torch.exp(-ik * (grid.x_left + grid.x_right)) * (ik * psi + dpsi) / (2 * ik)
        outs_a.append(a)
        outs_b.append(b)
    return torch.cat(outs_a, -1), torch.cat(outs_b, -1)


def _zakharov_shabat_ab_raw(q: Tensor, zeta: Tensor, grid: Grid) -> tuple[Tensor, Tensor]:
    """Jost coefficients a(zeta), b(zeta) of the Zakharov-Shabat system with
    focusing reduction r = -conj(q).  q: (..., n) complex; zeta: (K,) complex,
    or (..., K) matching q's batch shape.  Returns (..., K)."""
    cdt = complex_dtype(q.dtype)
    q = q.to(cdt)
    zeta = zeta.to(dtype=cdt, device=q.device)
    batch = q.shape[:-1]
    nb = int(torch.tensor(batch).prod().item()) if batch else 1
    outs_a, outs_b = [], []
    for sl in chunked(zeta.shape[-1], _chunk_size(grid.n, nb)):
        z = zeta[..., sl]                                    # (Kc,) or (..., Kc)
        qq = q[..., None, :]
        z_ = z[..., :, None]
        mu2 = -(z_ ** 2) - (qq.abs() ** 2)                   # (..., Kc, n)
        A = torch.empty(*mu2.shape, 2, 2, dtype=cdt, device=q.device)
        A[..., 0, 0] = -1j * z_
        A[..., 1, 1] = 1j * z_
        A[..., 0, 1] = qq
        A[..., 1, 0] = -qq.conj()
        T = ordered_product(cell_propagators(A, mu2, grid.h))
        a = T[..., 0, 0] * torch.exp(1j * z * grid.length)
        b = T[..., 1, 0] * torch.exp(-1j * z * (grid.x_left + grid.x_right))
        outs_a.append(a)
        outs_b.append(b)
    return torch.cat(outs_a, -1), torch.cat(outs_b, -1)


def _inv2(T: Tensor) -> Tensor:
    """Inverse of unimodular 2x2 matrices."""
    return torch.stack([torch.stack([T[..., 1, 1], -T[..., 0, 1]], -1),
                        torch.stack([-T[..., 1, 0], T[..., 0, 0]], -1)], -2)


def _match(left: Tensor, right: Tensor) -> Tensor:
    """b such that left = b * right (2-vectors on the last axis), least squares."""
    return (right * left).sum(-1) / (right * right).sum(-1)


def _split_index(grid: Grid, x_match: float) -> int:
    """Number of cells left of the matching point (kept away from the ends)."""
    m = int(round((x_match - grid.x_left) / grid.h))
    return min(max(m, 1), grid.n - 1)


def _schrodinger_bound_b_raw(u1: Tensor, k: Tensor, grid: Grid, x_match: Tensor) -> Tensor:
    """b(k) at bound states k = i kappa, computed stably by matching the left
    Jost solution (integrated rightwards) with the right Jost solution
    (integrated leftwards) at x_match, which should sit near the peak of the
    eigenfunction: each solution is then only ever integrated in its growing
    direction.  (Matching elsewhere fails for separated multi-solitons, where
    an eigenfunction is exponentially small in the other wells.)
    u1: (n,), k: (K,), x_match: (K,)."""
    cdt = complex_dtype(u1.dtype)
    k = k.to(cdt)
    mu2 = u1.to(cdt)[None, :] - (k ** 2)[:, None]
    A = torch.zeros(*mu2.shape, 2, 2, dtype=cdt, device=u1.device)
    A[..., 0, 1] = 1
    A[..., 1, 0] = mu2
    P = cell_propagators(A, mu2, grid.h)
    ik = 1j * k
    out = []
    for i in range(k.numel()):
        m = _split_index(grid, float(x_match[i]))
        TL = ordered_product(P[i, :m])
        TR = ordered_product(P[i, m:])
        v_minus = torch.exp(-ik[i] * grid.x_left) * (TL @ torch.stack([torch.ones_like(ik[i]), -ik[i]]))
        v_plus = torch.exp(ik[i] * grid.x_right) * (_inv2(TR) @ torch.stack([torch.ones_like(ik[i]), ik[i]]))
        out.append(_match(v_minus, v_plus))
    return torch.stack(out)


def _zakharov_shabat_bound_b_raw(q1: Tensor, zeta: Tensor, grid: Grid, x_match: Tensor) -> Tensor:
    """b(zeta_n) at ZS eigenvalues by matching phi (from the left) with psi
    (from the right) at x_match (eigenfunction peak).  q1: (n,), zeta: (K,)."""
    cdt = complex_dtype(q1.dtype)
    q1 = q1.to(cdt)
    zeta = zeta.to(cdt)
    z_ = zeta[:, None]
    qq = q1[None, :]
    mu2 = -(z_ ** 2) - qq.abs() ** 2
    A = torch.empty(*mu2.shape, 2, 2, dtype=cdt, device=q1.device)
    A[..., 0, 0] = -1j * z_
    A[..., 1, 1] = 1j * z_
    A[..., 0, 1] = qq.expand_as(z_ * qq)
    A[..., 1, 0] = -qq.conj().expand_as(z_ * qq)
    P = cell_propagators(A, mu2, grid.h)
    out = []
    for i in range(zeta.numel()):
        m = _split_index(grid, float(x_match[i]))
        TL = ordered_product(P[i, :m])
        TR = ordered_product(P[i, m:])
        phi = torch.exp(-1j * zeta[i] * grid.x_left) * TL[:, 0]
        psi = torch.exp(1j * zeta[i] * grid.x_right) * _inv2(TR)[:, 1]
        out.append(_match(phi, psi))
    return torch.stack(out)


def _subsample(v: Tensor, grid: Grid, max_points: int) -> tuple[Tensor, Tensor, float]:
    stride = max(1, math.ceil(grid.n / max_points))
    xs = grid.x_min + grid.h * torch.arange(0, grid.n, stride, dtype=torch.float64, device=v.device)
    return v[::stride], xs, grid.h * stride


def kdv_eigenfunction_peaks(u1: Tensor, grid: Grid, kappa: Tensor, max_points: int = 800) -> Tensor:
    """Approximate peak positions of the bound states with the given kappa,
    from eigenvectors of a coarse finite-difference Hamiltonian."""
    with torch.no_grad():
        us, xs, h = _subsample(u1.to(torch.float64), grid, max_points)
        n = us.numel()
        off = torch.full((n - 1,), -1.0 / h ** 2, dtype=torch.float64, device=u1.device)
        H = torch.diag(2.0 / h ** 2 + us) + torch.diag(off, 1) + torch.diag(off, -1)
        w, V = torch.linalg.eigh(H)
        j = torch.argmin((w[None, :] + kappa.detach().to(torch.float64)[:, None] ** 2).abs(), dim=1)
        return xs[torch.argmax(V[:, j].abs(), dim=0)]


def zs_eigenfunction_peaks(q1: Tensor, grid: Grid, zeta: Tensor, max_points: int = 512) -> Tensor:
    """Approximate peak positions of the ZS eigenfunctions for eigenvalues zeta."""
    with torch.no_grad():
        L, xs = _zs_collocation(q1, grid, max_points)
        w, V = torch.linalg.eig(L)
        n = xs.numel()
        j = torch.argmin((w[None, :] - zeta.detach().to(torch.complex128)[:, None]).abs(), dim=1)
        dens = V[:n, j].abs() ** 2 + V[n:, j].abs() ** 2
        return xs[torch.argmax(dens, dim=0)]


# ---------------------------------------------------------------------------
# Richardson extrapolation
# ---------------------------------------------------------------------------
# The exponential-midpoint scheme is symmetric, so its error expands in even
# powers of h.  Combining the grid with the every-other-sample grid (cells of
# width 2h centred on the odd samples) as (4 f_h - f_2h)/3 gives O(h^4).

def _coarse(v: Tensor, grid: Grid) -> tuple[Tensor, Grid]:
    m = (grid.n - 1) // 2
    return v[..., 1:2 * m:2], Grid(grid.x_min + grid.h, grid.x_min + (2 * m - 1) * grid.h, m)


def _richardson(fn, v: Tensor, k: Tensor, grid: Grid, richardson: bool, *extra):
    out = fn(v, k, grid, *extra)
    if not richardson or grid.n < 9:
        return out
    vc, gc = _coarse(v, grid)
    outc = fn(vc, k, gc, *extra)
    if isinstance(out, tuple):
        return tuple((4 * f - c) / 3 for f, c in zip(out, outc))
    return (4 * out - outc) / 3


def schrodinger_ab(u: Tensor, k: Tensor, grid: Grid, richardson: bool = True) -> tuple[Tensor, Tensor]:
    """Jost coefficients a(k), b(k) of -psi'' + u psi = k^2 psi.

    u: (..., n) real potential samples on `grid`; k: (K,) complex with
    Im k >= 0 (or real).  Returns a, b of shape (..., K).  O(h^4) with
    Richardson extrapolation (default), O(h^2) without."""
    return _richardson(_schrodinger_ab_raw, u, k, grid, richardson)


def zakharov_shabat_ab(q: Tensor, zeta: Tensor, grid: Grid, richardson: bool = True) -> tuple[Tensor, Tensor]:
    """Jost coefficients a(zeta), b(zeta) of the focusing Zakharov-Shabat
    system.  q: (..., n) complex; zeta: (K,) complex.  Returns (..., K)."""
    return _richardson(_zakharov_shabat_ab_raw, q, zeta, grid, richardson)


def schrodinger_bound_b(u1: Tensor, k: Tensor, grid: Grid, x_match: Tensor, richardson: bool = True) -> Tensor:
    return _richardson(_schrodinger_bound_b_raw, u1, k, grid, richardson, x_match)


def zakharov_shabat_bound_b(q1: Tensor, zeta: Tensor, grid: Grid, x_match: Tensor, richardson: bool = True) -> Tensor:
    return _richardson(_zakharov_shabat_bound_b_raw, q1, zeta, grid, richardson, x_match)


# ---------------------------------------------------------------------------
# KdV direct transform
# ---------------------------------------------------------------------------

def _kdv_ab_kappa(u1: Tensor, kappa: Tensor, grid: Grid) -> tuple[Tensor, Tensor]:
    a, b = schrodinger_ab(u1, 1j * kappa.to(complex_dtype(u1.dtype)), grid)
    return a.real, b.real


def _fd_derivative(f, z: Tensor, delta: float) -> Tensor:
    """4th-order central difference of a (holomorphic) function along the real direction."""
    return (-f(z + 2 * delta) + 8 * f(z + delta) - 8 * f(z - delta) + f(z - 2 * delta)) / (12 * delta)


def kdv_bound_states(u1: Tensor, grid: Grid, n_scan: int = 600, tol: float = 1e-13,
                     kappa_min: float = 1e-3) -> Tensor:
    """Locate the kappa_n (no grad) for a single potential u1: (n,).

    a(i kappa) is real on kappa > 0 and its zeros are simple and interlace,
    so a sign-change scan followed by safeguarded bisection/secant finds
    every bound state with kappa > kappa_min.
    """
    with torch.no_grad():
        umin = u1.min().item()
        if umin >= 0:
            return u1.new_zeros(0)
        kmax = math.sqrt(-umin) * 1.0001 + 1e-6
        ks = torch.linspace(kappa_min, kmax, n_scan, dtype=u1.dtype, device=u1.device)
        a, _ = _kdv_ab_kappa(u1, ks, grid)
        s = torch.sign(a)
        idx = torch.nonzero(s[:-1] * s[1:] < 0).flatten()
        if idx.numel() == 0:
            return u1.new_zeros(0)
        lo, hi = ks[idx].clone(), ks[idx + 1].clone()
        flo = a[idx].clone()
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            fm, _ = _kdv_ab_kappa(u1, mid, grid)
            left = torch.sign(fm) == torch.sign(flo)
            lo = torch.where(left, mid, lo)
            flo = torch.where(left, fm, flo)
            hi = torch.where(left, hi, mid)
            if (hi - lo).max().item() < tol:
                break
        return torch.flip(0.5 * (lo + hi), [0])  # largest kappa first


def kdv_direct(u: Tensor, x: Tensor, k: Tensor | None = None, *, reflection: bool = True,
               n_scan: int = 600) -> KdVScatteringData:
    """Direct scattering transform for KdV.

    u: (..., n) real potential on the uniform grid x (n,).
    k: real spectral grid for R(k); defaults to `default_spectral_grid`.
    Differentiable w.r.t. u (bound states via implicit differentiation).
    """
    grid = Grid.from_x(x)
    batch = u.shape[:-1]
    flat = u.reshape(-1, u.shape[-1])
    kappas, logcs = [], []
    delta = 1e-3
    for u1 in flat:
        k0 = kdv_bound_states(u1, grid, n_scan=n_scan)
        if k0.numel() == 0:
            kappas.append(k0)
            logcs.append(k0)
            continue
        f = lambda kk: _kdv_ab_kappa(u1, kk, grid)[0]  # noqa: E731
        d = min(delta, 0.25 * k0.min().item())
        a0 = f(k0)
        da0 = _fd_derivative(f, k0, d)
        # one differentiable Newton step: exact first-order sensitivity
        kap = k0 - a0 / da0.detach()
        a_k = _fd_derivative(f, kap, d)
        xm = kdv_eigenfunction_peaks(u1, grid, kap)
        b = schrodinger_bound_b(u1, 1j * kap.to(complex_dtype(u1.dtype)), grid, xm).real
        c2 = b / a_k
        logcs.append(0.5 * torch.log(c2.abs()))
        kappas.append(kap)
    nmax = max((kk.numel() for kk in kappas), default=0)
    kappa = u.new_ones(len(kappas), nmax)
    log_c = u.new_zeros(len(kappas), nmax)
    mask = torch.zeros(len(kappas), nmax, dtype=torch.bool, device=u.device)
    rows_k, rows_c = [], []
    for i, (kk, lc) in enumerate(zip(kappas, logcs)):
        m = kk.numel()
        rows_k.append(torch.cat([kk, u.new_ones(nmax - m)]))
        rows_c.append(torch.cat([lc, u.new_zeros(nmax - m)]))
        mask[i, :m] = True
    if nmax:
        kappa = torch.stack(rows_k)
        log_c = torch.stack(rows_c)
    kappa = kappa.reshape(*batch, nmax)
    log_c = log_c.reshape(*batch, nmax)
    mask = mask.reshape(*batch, nmax)

    R = None
    if reflection:
        if k is None:
            k = default_spectral_grid(grid, device=u.device, dtype=u.dtype)
        a, b = schrodinger_ab(u, k.to(complex_dtype(u.dtype)), grid)
        R = b / a
    return KdVScatteringData(kappa, log_c, mask, k if reflection else None, R)


# ---------------------------------------------------------------------------
# NLS direct transform
# ---------------------------------------------------------------------------

def _fourier_diff_matrix(n: int, h: float, dtype, device) -> Tensor:
    k = 2 * math.pi * torch.fft.fftfreq(n, d=h, device=device, dtype=torch.float64)
    if n % 2 == 0:
        k[n // 2] = 0.0  # drop the Nyquist mode for first derivatives
    eye = torch.eye(n, dtype=torch.complex128, device=device)
    D = torch.fft.ifft(1j * k[:, None] * torch.fft.fft(eye, dim=0), dim=0)
    return D.to(dtype)


def _zs_collocation(q1: Tensor, grid: Grid, max_points: int) -> tuple[Tensor, Tensor]:
    """Fourier-collocation matrix of the ZS eigenproblem
    zeta v = [[i d/dx, -i q], [-i conj(q), -i d/dx]] v  on a subsampled grid."""
    qs, xs, h = _subsample(q1.to(torch.complex128), grid, max_points)
    n = qs.numel()
    D = _fourier_diff_matrix(n, h, torch.complex128, q1.device)
    Q = torch.diag(qs)
    L = torch.cat([torch.cat([1j * D, -1j * Q], 1),
                   torch.cat([-1j * Q.conj(), -1j * D], 1)], 0)
    return L, xs


def zs_eigenvalue_candidates(q1: Tensor, grid: Grid, max_points: int = 512) -> Tensor:
    """Fourier-collocation eigenvalues of the ZS operator in the upper half plane
    (initial guesses only; refined by Newton on a(zeta))."""
    with torch.no_grad():
        L, xs = _zs_collocation(q1, grid, max_points)
        h = float(xs[1] - xs[0])
        ev = torch.linalg.eigvals(L)
        qmax = q1.abs().max().item()
        keep = (ev.imag > 1e-3 * max(qmax, 1e-12)) & (ev.abs() < 4 * qmax + math.pi / h * 0.5)
        return ev[keep]


def nls_eigenvalues(q1: Tensor, grid: Grid, max_points: int = 512, iters: int = 50,
                    tol: float = 1e-11) -> Tensor:
    """Discrete eigenvalues zeta_n (Im > 0, no grad) for one potential q1: (n,)."""
    with torch.no_grad():
        cand = zs_eigenvalue_candidates(q1, grid, max_points)
        if cand.numel() == 0:
            return cand
        f = lambda z: zakharov_shabat_ab(q1, z, grid)[0]  # noqa: E731
        z = cand.to(torch.complex128)
        for _ in range(iters):
            a = f(z)
            da = _fd_derivative(f, z, 1e-4)
            step = a / da
            step = torch.where(step.abs() > 0.5, 0.5 * step / step.abs(), step)
            z = z - step
            if step.abs().max().item() < tol:
                break
        a = f(z)
        good = (z.imag > 1e-6) & (a.abs() < 1e-8) & torch.isfinite(z)
        z = z[good]
        # deduplicate
        out: list[complex] = []
        for zz in z.tolist():
            if all(abs(zz - o) > 1e-6 for o in out):
                out.append(zz)
        out.sort(key=lambda v: -v.imag)
        return torch.tensor(out, dtype=torch.complex128, device=q1.device)


def nls_direct(q: Tensor, x: Tensor, xi: Tensor | None = None, *, reflection: bool = True,
               max_points: int = 512) -> NLSScatteringData:
    """Direct scattering transform for the focusing NLS (Zakharov-Shabat).

    q: (..., n) complex (or real) potential on the uniform grid x.
    """
    grid = Grid.from_x(x)
    cdt = complex_dtype(q.dtype)
    q = q.to(cdt)
    batch = q.shape[:-1]
    flat = q.reshape(-1, q.shape[-1])
    zetas, logCs = [], []
    for q1 in flat:
        z0 = nls_eigenvalues(q1, grid, max_points).to(cdt)
        if z0.numel() == 0:
            zetas.append(z0)
            logCs.append(z0)
            continue
        f = lambda z: zakharov_shabat_ab(q1, z, grid)[0]  # noqa: E731
        d = 1e-3 * min(1.0, z0.imag.min().item())
        da0 = _fd_derivative(f, z0, d)
        z = z0 - f(z0) / da0.detach()
        a_z = _fd_derivative(f, z, d)
        xm = zs_eigenfunction_peaks(q1, grid, z)
        b = zakharov_shabat_bound_b(q1, z, grid, xm)
        zetas.append(z)
        logCs.append(torch.log(b / a_z))
    nmax = max((z.numel() for z in zetas), default=0)
    pad_z = torch.complex(torch.zeros((), dtype=x.dtype), torch.ones((), dtype=x.dtype)).to(cdt).to(q.device)
    rows_z, rows_c = [], []
    mask = torch.zeros(len(zetas), nmax, dtype=torch.bool, device=q.device)
    for i, (z, lc) in enumerate(zip(zetas, logCs)):
        m = z.numel()
        rows_z.append(torch.cat([z, pad_z.expand(nmax - m)]))
        rows_c.append(torch.cat([lc, torch.zeros(nmax - m, dtype=cdt, device=q.device)]))
        mask[i, :m] = True
    zeta = torch.stack(rows_z).reshape(*batch, nmax)
    log_C = torch.stack(rows_c).reshape(*batch, nmax)
    mask = mask.reshape(*batch, nmax)

    r = None
    if reflection:
        if xi is None:
            xi = default_spectral_grid(grid, device=q.device, dtype=x.dtype)
        a, b = zakharov_shabat_ab(q, xi.to(cdt), grid)
        r = b / a
    return NLSScatteringData(zeta, log_C, mask, xi if reflection else None, r)
