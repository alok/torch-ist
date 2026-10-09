"""Inverse scattering: Gel'fand-Levitan-Marchenko equations.

KdV (right Marchenko equation, x < y):
    K(x,y) + F(x+y) + int_x^inf K(x,z) F(z+y) dz = 0,
    F(s) = sum_n c_n^2 e^{-kappa_n s} + (1/2pi) int R(k) e^{iks} dk,
    u(x) = -2 d/dx K(x,x).

Focusing NLS (Zakharov-Shabat, r = -conj(q)); with G = conj(K_1):
    G(x,y) = F(x+y) + int_x^inf L(x,s) F(s+y) ds,
    L(x,y) = -int_x^inf G(x,s) conj(F(s+y)) ds,
    F(s) = (1/2pi) int r(xi) e^{i xi s} dxi - i sum_n C_n e^{i zeta_n s},
    q(x) = -2 conj(G(x,x)).

Numerics
--------
The soliton part of F is a degenerate (rank-N) kernel whose size grows like
e^{2 kappa |x - x_n|} away from soliton n -- a naive discretisation loses
all accuracy a few widths from each soliton.  We split
    (I + F_c + U Lambda U^T)
into the well-conditioned continuous part and a low-rank correction handled by
a scaled Woodbury identity (docs/DERIVATIONS.md):

    K(x,x) = beta + (rho - a) * Y * gamma,
    Y = D (I + P D)^{-1} = A1 (S^{-2} + A2 P A1)^{-1} A2,

where D = S A1 A2 S with S >= 1 carrying the huge magnitudes and A1, A2
bounded.  Every matrix that is actually formed has O(1) entries, so N-soliton
data of any separation reconstructs to machine precision, and the formula is
smooth (autograd friendly).  With R = 0 this collapses to the classical
Kay-Moses / N-soliton determinant formula in closed form.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .direct import KdVScatteringData, NLSScatteringData
from .utils import Grid, chunked, complex_dtype, quadrature_weights

_NEG = -1e4  # log-magnitude used for padded (masked) solitons: exp(_NEG) == 0


def _scale(log_abs: Tensor) -> tuple[Tensor, Tensor]:
    """Split |lambda| = sigma * |phi| with sigma = max(|lambda|, 1).

    Returns (1/sigma, |phi|), both in [0, 1]."""
    log_sigma = torch.clamp(log_abs, min=0.0)
    return torch.exp(-log_sigma), torch.exp(log_abs - log_sigma)


# ---------------------------------------------------------------------------
# Reflectionless (closed form)
# ---------------------------------------------------------------------------

def _kdv_Kxx(kappa: Tensor, log_c: Tensor, mask: Tensor, x: Tensor) -> Tensor:
    """K(x,x) = d/dx log det(I + A) for reflectionless KdV data.

    kappa, log_c, mask: (..., N); x: (..., Nx) -> (..., Nx)."""
    log_lam = 2 * log_c[..., None, :] - 2 * kappa[..., None, :] * x[..., :, None]  # (..., Nx, N)
    log_lam = torch.where(mask[..., None, :], log_lam, torch.full_like(log_lam, _NEG))
    s_inv, phi = _scale(log_lam)
    P = 1.0 / (kappa[..., :, None] + kappa[..., None, :])                       # (..., N, N)
    M = torch.diag_embed(s_inv) + phi[..., :, None] * P[..., None, :, :]
    y = torch.linalg.solve(M, phi)                                               # Y @ 1
    return -y.sum(-1)


def kdv_reflectionless(data: KdVScatteringData, x: Tensor) -> Tensor:
    """N-soliton KdV potential u(x) = -2 d^2/dx^2 log det(I + A(x)),
    A_mn = c_m c_n e^{-(kappa_m + kappa_n) x}/(kappa_m + kappa_n).

    Fully differentiable in (kappa, log_c) (and x).  x: (Nx,) any points.
    Returns (..., Nx)."""
    batch = data.kappa.shape[:-1]
    if data.kappa.shape[-1] == 0:
        return x.new_zeros(*batch, x.shape[-1])
    need_graph = torch.is_grad_enabled() and (data.kappa.requires_grad or data.log_c.requires_grad)
    with torch.enable_grad():
        xx = x.detach().to(data.kappa.dtype).expand(*batch, x.shape[-1]).clone().requires_grad_(True)
        K = _kdv_Kxx(data.kappa, data.log_c, data.mask, xx)
        (dK,) = torch.autograd.grad(K.sum(), xx, create_graph=need_graph)
    return -2 * dK


def _nls_lambda(zeta: Tensor, log_C: Tensor, mask: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    """lambda_n(x) = -i C_n e^{2 i zeta_n x}, returned as (log|lambda|, e^{i arg lambda})."""
    log_lam = log_C[..., None, :] - 0.5j * math.pi + 2j * zeta[..., None, :] * x[..., :, None].to(zeta.dtype)
    log_abs = torch.where(mask[..., None, :], log_lam.real, torch.full_like(log_lam.real, _NEG))
    return log_abs, torch.exp(1j * log_lam.imag)


def _nls_Y_gamma(P: Tensor, log_abs: Tensor, phase: Tensor, gamma: Tensor) -> Tensor:
    """Y gamma for D = [[0, conj(Lam)], [-Lam, 0]] via the scaled Woodbury form
    Y = J (S^{-2} + A2 P J)^{-1} A2,  J = [[0, I], [-I, 0]],  A2 = diag(Phi, conj(Phi))."""
    N = log_abs.shape[-1]
    s_inv, phi_abs = _scale(log_abs)
    phi = phi_abs * phase
    A2 = torch.cat([phi, phi.conj()], -1)                                  # (..., 2N)
    S = torch.cat([s_inv, s_inv], -1).to(phi.dtype)
    PJ = torch.cat([-P[..., N:], P[..., :N]], -1)                          # P @ J (column permutation)
    M = torch.diag_embed(S) + A2[..., :, None] * PJ
    v = torch.linalg.solve(M, A2 * gamma)
    return torch.cat([v[..., N:], -v[..., :N]], -1)                        # J v


def nls_reflectionless(data: NLSScatteringData, x: Tensor) -> Tensor:
    """N-soliton (reflectionless) focusing-NLS potential q(x).

    Closed form of the Marchenko equation with degenerate kernel, evaluated
    in a scaled form that never overflows.  Differentiable in (zeta, log_C).
    x: (Nx,) -> (..., Nx) complex."""
    zeta, log_C, mask = data.zeta, data.log_C, data.mask
    batch = zeta.shape[:-1]
    N = zeta.shape[-1]
    if N == 0:
        return torch.zeros(*batch, x.shape[-1], dtype=complex_dtype(x.dtype), device=x.device)
    log_abs, phase = _nls_lambda(zeta, log_C, mask, x)                     # (..., Nx, N)
    Gam = 1j / (zeta[..., :, None] - zeta[..., None, :].conj())             # (..., N, N)
    Z = torch.zeros_like(Gam)
    P = torch.cat([torch.cat([Gam, Z], -1), torch.cat([Z, Gam.conj()], -1)], -2)[..., None, :, :]
    gamma = torch.cat([torch.ones_like(phase), torch.zeros_like(phase)], -1)
    Yg = _nls_Y_gamma(P, log_abs, phase, gamma)
    G = -Yg[..., N:].sum(-1)                                                # rho = [0, -1]
    return -2 * G.conj()


# ---------------------------------------------------------------------------
# General data: discretised Marchenko + low-rank soliton correction
# ---------------------------------------------------------------------------

def _continuous_kernel(spec: Tensor, refl: Tensor, s: Tensor) -> Tensor:
    """F_c(s) = (1/2pi) int refl(k) e^{iks} dk by the trapezoid rule.

    spec: (K,) sorted; refl: (..., K); s: (S,) -> (..., S) complex."""
    w = quadrature_weights(spec).to(refl.dtype) / (2 * math.pi)
    rw = refl * w
    out = []
    for sl in chunked(s.numel(), max(1, 4_000_000 // spec.numel())):
        E = torch.exp(1j * s[sl, None].to(refl.dtype) * spec[None, :].to(refl.dtype))  # (Sc, K)
        out.append(rw @ E.transpose(0, 1))
    return torch.cat(out, -1)


def _windows(grid: Grid, idx: Tensor, dtype, device) -> tuple[Tensor, Tensor, int]:
    """Local quadrature windows for the Marchenko integral at x_i, i in idx.

    The kernel F(z+y) is O(1) wherever z + y lies left of the potential, so
    the integral over z in [x_i, inf) must run to z = 2 x_max - x_i (beyond
    which F(z+y) = 0 for every y >= x_i), not just to the grid edge.
    Local nodes z_l = x_i + l h, l < W_i = 2(n-1-i) + 1.  idx must be
    increasing; returns sqrt trapezoid weights (C, W), validity (C, W), W."""
    W = 2 * (grid.n - 1 - int(idx[0])) + 1
    l = torch.arange(W, device=device)
    last = 2 * (grid.n - 1 - idx)                                              # (C,)
    valid = l[None, :] <= last[:, None]
    # Gregory end corrections (3rd-order trapezoid) at both ends of each window
    g = torch.tensor([3 / 8, 7 / 6, 23 / 24], dtype=dtype, device=device)
    w = torch.ones(len(idx), W, dtype=dtype, device=device)
    for j in range(3):
        w[:, j] = g[j]
        w = torch.where(l[None, :] == (last - j)[:, None], g[j], w)
    w = torch.where(valid, w * grid.h, torch.zeros_like(w))
    return torch.sqrt(w), valid, W


def _chunk_for(W: int, cplx: bool, budget: float = 4e8) -> int:
    per = W * W * (16 if cplx else 8) * 4
    return max(1, int(budget // per))


def _march_chunks(n: int, cplx: bool, chunk: int | None):
    i = 0
    while i < n:
        W = 2 * (n - 1 - i) + 1
        c = chunk or _chunk_for(W, cplx)
        yield torch.arange(i, min(i + c, n))
        i += c


def _local_hankel(Fc: Tensor, idx: Tensor, W: int) -> tuple[Tensor, Tensor]:
    """H[c, l1, l2] = Fc[2 i_c + l1 + l2] and f[c, l] = Fc[2 i_c + l]."""
    l = torch.arange(W, device=Fc.device)
    top = Fc.shape[-1] - 1
    hidx = torch.clamp(2 * idx[:, None, None] + l[None, :, None] + l[None, None, :], max=top)
    fidx = torch.clamp(2 * idx[:, None] + l[None, :], max=top)
    return Fc[..., hidx], Fc[..., fidx]


def _ddx(f: Tensor, h: float) -> Tensor:
    """4th-order central first derivative along the last axis (2nd order at the edges)."""
    d = torch.gradient(f, spacing=h, dim=-1, edge_order=2)[0]
    if f.shape[-1] >= 5:
        d[..., 2:-2] = (f[..., :-4] - 8 * f[..., 1:-3] + 8 * f[..., 3:-1] - f[..., 4:]) / (12 * h)
    return d


def _s_grid(grid: Grid, dtype, device) -> Tensor:
    # s = z + y ranges over [2 x_min, 2 (2 x_max - x_min)]
    return 2 * grid.x_min + grid.h * torch.arange(4 * grid.n - 3, device=device, dtype=dtype)


def kdv_marchenko(data: KdVScatteringData, x: Tensor, chunk: int | None = None) -> Tensor:
    """Solve the KdV Marchenko equation on the uniform grid x and return u(x).

    Uses data.R (continuous spectrum) and the bound states (low-rank part).
    Cost ~2 n^4 flops for n grid points -- use n of a few hundred."""
    grid = Grid.from_x(x)
    n, h = grid.n, grid.h
    dt = x.dtype
    dev = x.device
    batch = data.kappa.shape[:-1]
    Ns = data.kappa.shape[-1]
    s = _s_grid(grid, dt, dev)
    if data.R is not None:
        Fc = _continuous_kernel(data.k.to(dev), data.R, s).real
    else:
        Fc = torch.zeros(*batch, s.numel(), dtype=dt, device=dev)
    Kxx = []
    for idx in _march_chunks(n, False, chunk):
        sw, valid, W = _windows(grid, idx, dt, dev)                            # (C, W)
        Hl, fl = _local_hankel(Fc, idx, W)                                     # (..., C, W, W), (..., C, W)
        B = torch.eye(W, dtype=dt, device=dev) + sw[:, :, None] * Hl * sw[:, None, :]
        fc = sw * fl
        e0 = torch.zeros(W, dtype=dt, device=dev)
        e0[0] = 1
        rhs = [e0.expand(*B.shape[:-2], W)[..., None]]
        if Ns:
            z = h * torch.arange(W, device=dev, dtype=dt)
            U = sw[..., None] * torch.exp(-data.kappa[..., None, None, :] * z[:, None])  # (..., C, W, Ns)
            rhs.append(U)
        Z = torch.linalg.solve(B, torch.cat(rhs, -1))
        Z0 = Z[..., 0]
        K = -(fc * Z0).sum(-1)
        if Ns:
            ZU = Z[..., 1:]
            a = (fc[..., None] * ZU).sum(-2)                                   # (..., C, Ns)
            # P = U^T B^{-1} U = (exact Gram integral) + U^T (B^{-1} - I) U: the
            # identity part is integrated analytically, only the radiation
            # coupling is left to the quadrature
            ksum = data.kappa[..., None, :, None] + data.kappa[..., None, None, :]  # (..., 1, Ns, Ns)
            Zend = (h * (2 * (n - 1 - idx))).to(dt)[:, None, None]                 # window length (C,1,1)
            gram = -torch.expm1(-ksum * Zend) / ksum
            P = U.transpose(-1, -2) @ (ZU - U) + gram                         # (..., C, Ns, Ns)
            gamma = (U * Z0[..., None]).sum(-2)                                # (..., C, Ns)
            log_lam = 2 * data.log_c[..., None, :] - 2 * data.kappa[..., None, :] * x[idx][:, None]
            log_lam = torch.where(data.mask[..., None, :], log_lam, torch.full_like(log_lam, _NEG))
            s_inv, phi = _scale(log_lam)
            M = torch.diag_embed(s_inv) + phi[..., :, None] * P
            Yg = torch.linalg.solve(M, phi * gamma)
            K = K - ((1 - a) * Yg).sum(-1)
        Kxx.append(K / sw[:, 0])
    Kxx = torch.cat(Kxx, -1)
    return -2 * _ddx(Kxx, h)


def nls_marchenko(data: NLSScatteringData, x: Tensor, chunk: int | None = None) -> Tensor:
    """Solve the focusing-NLS Marchenko system on the uniform grid x; returns q(x)."""
    grid = Grid.from_x(x)
    n, h = grid.n, grid.h
    rdt = x.dtype
    cdt = complex_dtype(rdt)
    dev = x.device
    batch = data.zeta.shape[:-1]
    Ns = data.zeta.shape[-1]
    s = _s_grid(grid, rdt, dev)
    if data.r is not None:
        Fc = _continuous_kernel(data.xi.to(dev), data.r, s)
    else:
        Fc = torch.zeros(*batch, s.numel(), dtype=cdt, device=dev)
    out = []
    for idx in _march_chunks(n, True, chunk):
        sw, valid, W = _windows(grid, idx, rdt, dev)
        swc = sw.to(cdt)
        Hl, fl = _local_hankel(Fc, idx, W)
        Ht = swc[:, :, None] * Hl * swc[:, None, :]                            # (..., C, W, W)
        Htb = Ht.conj()
        Sm = torch.eye(W, dtype=cdt, device=dev) + Htb @ Ht                    # I + conj(H) H
        fc = swc * fl
        e0 = torch.zeros(W, dtype=cdt, device=dev)
        e0[0] = 1
        e0 = e0.expand(*Ht.shape[:-2], W)
        # columns [r1; r2] to which M_c^{-1} is applied, M_c = [[I, conj(H)], [-H, I]]
        r1 = [e0[..., None]]
        r2 = [torch.zeros_like(e0)[..., None]]
        U = Ub = None
        if Ns:
            z = (h * torch.arange(W, device=dev, dtype=rdt)).to(cdt)
            U = swc[..., None] * torch.exp(1j * data.zeta[..., None, None, :] * z[:, None])  # (..., C, W, Ns)
            Ub = U.conj()
            zero = torch.zeros_like(U)
            # calU = blockdiag(conj U, U)
            r1 += [Ub, zero]
            r2 += [zero, U]
        R1 = torch.cat(r1, -1)
        R2 = torch.cat(r2, -1)
        Z1 = torch.linalg.solve(Sm, R1 - Htb @ R2)
        Z2 = R2 + Ht @ Z1
        G = (fc * Z1[..., 0]).sum(-1)
        if Ns:
            a = (fc[..., None] * Z1[..., 1:]).sum(-2)                          # f_c M^{-1} calU
            VtZ = torch.cat([U.transpose(-1, -2) @ Z1, Ub.transpose(-1, -2) @ Z2], -2)  # calV^T M^{-1} [e0, calU]
            gamma = VtZ[..., 0]
            # replace the quadrature of calV^T calU (the identity part of M_c)
            # by the exact Gram integrals int_0^Z e^{i (zeta_m - conj zeta_n) z} dz
            zero_ns = torch.zeros_like(U)
            calU = torch.cat([torch.cat([Ub, zero_ns], -1), torch.cat([zero_ns, U], -1)], -2)
            VtU = torch.cat([U.transpose(-1, -2) @ calU[..., :W, :], Ub.transpose(-1, -2) @ calU[..., W:, :]], -2)
            dzeta = data.zeta[..., None, :, None] - data.zeta[..., None, None, :].conj()  # (..., 1, Ns, Ns)
            Zend = (h * (2 * (n - 1 - idx))).to(cdt)[:, None, None]
            Gam = torch.expm1(1j * dzeta * Zend) / (1j * dzeta)
            Zb = torch.zeros_like(Gam)
            gram = torch.cat([torch.cat([Gam, Zb], -1), torch.cat([Zb, Gam.conj()], -1)], -2)
            P = VtZ[..., 1:] - VtU + gram
            log_abs, phase = _nls_lambda(data.zeta, data.log_C, data.mask, x[idx])
            Yg = _nls_Y_gamma(P, log_abs, phase, gamma)
            rho = torch.cat([torch.zeros(Ns, dtype=cdt, device=dev), -torch.ones(Ns, dtype=cdt, device=dev)])
            G = G + ((rho - a) * Yg).sum(-1)
        out.append(G / swc[:, 0])
    G = torch.cat(out, -1)
    return -2 * G.conj()


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

def kdv_inverse(data: KdVScatteringData, x: Tensor, method: str = "auto") -> Tensor:
    """Reconstruct u(x).  method: 'auto' | 'closed_form' | 'marchenko'.
    'auto' uses the closed form when there is no reflection coefficient."""
    if method == "closed_form" or (method == "auto" and data.R is None):
        return kdv_reflectionless(data, x)
    return kdv_marchenko(data, x)


def nls_inverse(data: NLSScatteringData, x: Tensor, method: str = "auto") -> Tensor:
    """Reconstruct q(x).  method: 'auto' | 'closed_form' | 'marchenko'."""
    if method == "closed_form" or (method == "auto" and data.r is None):
        return nls_reflectionless(data, x)
    return nls_marchenko(data, x)
