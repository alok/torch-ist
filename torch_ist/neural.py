"""Learnable inverse scattering: a neural operator whose spectral domain is
scattering space.

A Fourier neural operator (FNO) learns in the domain where *linear* PDEs
diagonalise.  For integrable nonlinear PDEs the inverse scattering transform
plays the same role: in scattering coordinates the KdV flow is

    kappa_n  -> kappa_n                (fixed)
    x0_n     -> x0_n + 4 kappa_n^2 t   (each soliton just translates)
    R(k)     -> R(k) e^{8 i k^3 t}     (pure phase rotation)

so the only hard parts are the two transforms.  `ScatteringNeuralOperator`
composes

    u(x,0) --[encoder: exact | learned]--> scattering data
           --[exact evolution]----------> scattering data at t
           --[decoder: exact | learned]--> u(x,t)

and any of the four combinations can be selected at run time.

Scattering data are represented as N fixed *slots* (kappa, x0, presence)
plus an optional reflection coefficient on a fixed k-grid.  The position
parameter x0 is defined by c^2 = 2 kappa e^{2 kappa x0}: for an isolated
soliton it is its centre, which keeps the regression targets O(domain size).

Contents
--------
SolitonSlots                 slot representation + exact evolution + conversions
ScatteringEncoder            1-D CNN: u(x) -> slots (+ R)
ScatteringDecoder            translation-equivariant renderer + dilated CNN: slots -> u(x)
ScatteringNeuralOperator     the end-to-end operator with exact/learned switches
FNO1dBaseline                standard FNO taking (u0, t) -> u(x,t), for comparison
generate_kdv_soliton_data / generate_kdv_general_data / exact_kdv_solution
potential_loss / scattering_loss / kdv_residual_loss
train / train_baseline
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .direct import KdVScatteringData, kdv_direct
from .inverse import kdv_marchenko, kdv_reflectionless
from .utils import complex_dtype, quadrature_weights

Mode = Literal["exact", "learned"]


# ---------------------------------------------------------------------------
# Slot representation of KdV scattering data
# ---------------------------------------------------------------------------

@dataclass
class SolitonSlots:
    """KdV scattering data in a fixed-size, network-friendly form.

    kappa, x0, presence : (..., N).  presence is 0/1 for exact data and a
        probability for predicted data.  Slots are ordered by kappa (desc).
    logits : (..., N) presence logits (predicted data only, for the loss).
    R : (..., K) complex reflection coefficient on the grid k (K,), optional.
    """

    kappa: Tensor
    x0: Tensor
    presence: Tensor
    logits: Tensor | None = None
    R: Tensor | None = None
    k: Tensor | None = None

    # -- exact dynamics --------------------------------------------------
    def evolve(self, t: Tensor) -> "SolitonSlots":
        """Exact KdV evolution.  t: scalar, (Nt,) or broadcastable to the batch.

        With t of shape (Nt,) a time axis is inserted: (..., N) -> (..., Nt, N).
        """
        t = torch.as_tensor(t, dtype=self.kappa.dtype, device=self.kappa.device)
        if t.ndim == 1:
            def ins(v: Tensor | None) -> Tensor | None:
                return None if v is None else v.unsqueeze(-2).expand(*v.shape[:-1], t.shape[0], v.shape[-1])
            kappa, x0, pres, logits, R = map(ins, (self.kappa, self.x0, self.presence, self.logits, self.R))
            tt = t[:, None]
        else:
            kappa, x0, pres, logits, R = self.kappa, self.x0, self.presence, self.logits, self.R
            tt = t[..., None] if t.ndim else t
        x0 = x0 + 4 * kappa ** 2 * tt
        if R is not None:
            R = R * torch.exp(8j * self.k.to(R.dtype) ** 3 * tt.to(R.dtype))
        return SolitonSlots(kappa, x0, pres, logits, R, self.k)

    # -- conversions -------------------------------------------------------
    @staticmethod
    def from_scattering(data: KdVScatteringData, n_slots: int) -> "SolitonSlots":
        """Exact data -> slots (largest kappa first; truncated/padded to n_slots)."""
        kappa, log_c, mask = data.kappa, data.log_c, data.mask
        N = kappa.shape[-1]
        if N < n_slots:
            pad = n_slots - N
            kappa = F.pad(kappa, (0, pad), value=1.0)
            log_c = F.pad(log_c, (0, pad), value=0.0)
            mask = F.pad(mask, (0, pad), value=False)
        kappa, log_c, mask = kappa[..., :n_slots], log_c[..., :n_slots], mask[..., :n_slots]
        x0 = (log_c - 0.5 * torch.log(2 * kappa)) / kappa
        x0 = torch.where(mask, x0, torch.zeros_like(x0))
        return SolitonSlots(kappa, x0, mask.to(kappa.dtype), None, data.R, data.k)

    def to_scattering(self, threshold: float = 0.5) -> KdVScatteringData:
        """Slots -> exact-solver data (presence thresholded)."""
        kappa = self.kappa.clamp_min(1e-3)
        mask = self.presence > threshold
        log_c = 0.5 * torch.log(2 * kappa) + kappa * self.x0
        return KdVScatteringData(kappa, log_c, mask, self.k, self.R)

    def to(self, *args, **kwargs) -> "SolitonSlots":
        def mv(v):
            if v is None:
                return None
            if v.is_complex():
                a2 = [complex_dtype(a) if isinstance(a, torch.dtype) else a for a in args]
                return v.to(*a2, **kwargs)
            return v.to(*args, **kwargs)
        return SolitonSlots(*(mv(v) for v in (self.kappa, self.x0, self.presence, self.logits, self.R, self.k)))

    def flatten(self) -> tuple["SolitonSlots", tuple[int, ...]]:
        lead = self.kappa.shape[:-1]
        f = lambda v: None if v is None else v.reshape(-1, v.shape[-1])  # noqa: E731
        return SolitonSlots(f(self.kappa), f(self.x0), f(self.presence), f(self.logits), f(self.R), self.k), lead

    def __getitem__(self, idx) -> "SolitonSlots":
        g = lambda v: None if v is None else v[idx]  # noqa: E731
        return SolitonSlots(g(self.kappa), g(self.x0), g(self.presence), g(self.logits), g(self.R), self.k)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class _ChannelNorm(nn.Module):
    """LayerNorm over channels at each position: unlike GroupNorm/BatchNorm it
    uses no statistics over x, so it preserves translation equivariance."""

    def __init__(self, width: int):
        super().__init__()
        self.ln = nn.LayerNorm(width)

    def forward(self, h: Tensor) -> Tensor:
        return self.ln(h.transpose(1, 2)).transpose(1, 2)


class _ResBlock(nn.Module):
    def __init__(self, width: int, dilation: int, kernel: int = 5):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.c1 = nn.Conv1d(width, width, kernel, padding=pad, dilation=dilation)
        self.c2 = nn.Conv1d(width, width, kernel, padding=pad, dilation=dilation)
        self.norm = _ChannelNorm(width)

    def forward(self, h: Tensor) -> Tensor:
        return h + self.c2(F.gelu(self.c1(F.gelu(self.norm(h)))))


def _param_dtype(m: nn.Module) -> torch.dtype:
    return next(m.parameters()).dtype


class ScatteringEncoder(nn.Module):
    """Learned direct scattering transform  u(x) -> SolitonSlots.

    A strided 1-D CNN (inputs: u and a normalised coordinate channel, since
    soliton positions are absolute) followed by an MLP with three heads:
    presence logits, kappa (softplus) and x0 per slot, plus optionally the
    reflection coefficient on a fixed k-grid (real and imaginary parts).
    """

    def __init__(self, x: Tensor, n_slots: int = 4, width: int = 48, n_down: int = 4,
                 hidden: int = 256, k: Tensor | None = None, u_scale: float = 2.0):
        super().__init__()
        self.register_buffer("x", x.to(torch.get_default_dtype()))
        self.n_slots = n_slots
        self.x_scale = float(x.abs().max())
        self.u_scale = u_scale
        self.register_buffer("k", None if k is None else k.to(torch.get_default_dtype()))
        layers: list[nn.Module] = [nn.Conv1d(2, width, 5, padding=2)]
        n = x.numel()
        for _ in range(n_down):
            layers += [_ResBlock(width, 1), _ResBlock(width, 2),
                       nn.Conv1d(width, width, 4, stride=2, padding=1)]
            n = n // 2
        self.cnn = nn.Sequential(*layers)
        n_r = 0 if k is None else 2 * k.numel()
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(width * n, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 3 * n_slots + n_r))

    def forward(self, u0: Tensor) -> SolitonSlots:
        lead = u0.shape[:-1]
        u = u0.reshape(-1, u0.shape[-1]).to(_param_dtype(self))
        xc = (self.x / self.x_scale).to(u.dtype).expand_as(u)
        h = self.cnn(torch.stack([u / self.u_scale, xc], 1))
        out = self.mlp(h)
        N = self.n_slots
        logits = out[:, :N]
        kappa = F.softplus(out[:, N:2 * N]) + 0.05
        x0 = self.x_scale * out[:, 2 * N:3 * N]
        R = None
        if self.k is not None:
            rr = out[:, 3 * N:].reshape(-1, 2, self.k.numel())
            R = torch.complex(rr[:, 0], rr[:, 1])
            R = R.reshape(*lead, -1)
        rs = lambda v: v.reshape(*lead, N)  # noqa: E731
        return SolitonSlots(rs(kappa), rs(x0), torch.sigmoid(rs(logits)), rs(logits), R, self.k)


class ScatteringDecoder(nn.Module):
    """Learned inverse scattering transform  SolitonSlots -> u(x).

    1. Renderer: each slot is drawn by a shared MLP
           f_n(x) = kappa_n^2 * g_theta(kappa_n, kappa_n (x - x0_n)),
       which is translation-equivariant and respects the KdV scaling symmetry
       u -> kappa^2 U(kappa x) by construction; g_theta is learned.
    2. Interaction network: a dilated residual CNN *without* a coordinate
       channel (so the whole decoder is translation-equivariant) reads the
       individual weighted slot fields, their sum, and -- if radiation is
       present -- the exact linear Born feature F_c(2x) = (1/2pi) int R e^{2ikx} dk,
       and outputs a correction to the summed field.
    """

    def __init__(self, x: Tensor, n_slots: int = 4, width: int = 48,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
                 hidden: int = 64, k: Tensor | None = None):
        super().__init__()
        self.register_buffer("x", x.to(torch.get_default_dtype()))
        self.n_slots = n_slots
        self.render = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                    nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.use_radiation = k is not None
        if k is not None:
            # Born feature F_c(2x) as a fixed linear map of R (real, since R(-k) = conj R(k))
            w = quadrature_weights(k.double()) / (2 * math.pi)
            E = torch.exp(2j * x.double()[:, None] * k.double()[None, :]) * w
            self.register_buffer("born_re", E.real.to(torch.get_default_dtype()))
            self.register_buffer("born_im", E.imag.to(torch.get_default_dtype()))
        c_in = n_slots + 1 + (1 if k is not None else 0)
        self.lift = nn.Conv1d(c_in, width, 5, padding=2)
        self.blocks = nn.Sequential(*[_ResBlock(width, d) for d in dilations])
        self.proj = nn.Conv1d(width, 1, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def slot_fields(self, kappa: Tensor, x0: Tensor) -> Tensor:
        """(B, N) -> (B, N, n_x) per-slot rendered profiles."""
        xi = kappa[..., None] * (self.x.to(kappa.dtype) - x0[..., None])
        kk = kappa[..., None].expand_as(xi)
        feats = torch.stack([kk, xi, torch.tanh(xi)], -1)
        return kappa[..., None] ** 2 * self.render(feats)[..., 0]

    def forward(self, slots: SolitonSlots) -> Tensor:
        s, lead = slots.flatten()
        dt = _param_dtype(self)
        kappa, x0, p = s.kappa.to(dt), s.x0.to(dt), s.presence.to(dt)
        fields = p[..., None] * self.slot_fields(kappa, x0)              # (B, N, n_x)
        base = fields.sum(1, keepdim=True)
        chans = [base, fields]
        if self.use_radiation:
            if s.R is None:
                chans.append(torch.zeros_like(base))
            else:
                R = s.R
                born = (R.real.to(dt) @ self.born_re.T.to(dt) - R.imag.to(dt) @ self.born_im.T.to(dt))  # Re(E R)
                chans.append(born[:, None, :])
        h = self.blocks(self.lift(torch.cat(chans, 1)))
        u = base[:, 0] + self.proj(h)[:, 0]
        return u.reshape(*lead, -1)


class ScatteringNeuralOperator(nn.Module):
    """u(x,0) -> u(x,t) through scattering space with exact time evolution.

    forward_mode / inverse_mode in {"exact", "learned"} select each transform:
        exact/exact     classical IST (reference)
        exact/learned   exact direct transform + learned inverse
        learned/exact   learned direct transform + exact (closed form) inverse
        learned/learned fully neural; the evolution step is still exact
    """

    def __init__(self, x: Tensor, n_slots: int = 4, encoder: ScatteringEncoder | None = None,
                 decoder: ScatteringDecoder | None = None, forward_mode: Mode = "learned",
                 inverse_mode: Mode = "learned", k: Tensor | None = None):
        super().__init__()
        # exact transforms run on CPU in float64; rebuild the grid from its
        # endpoints so a float32 grid does not turn into a non-uniform one
        self.x64 = torch.linspace(float(x[0]), float(x[-1]), x.numel(), dtype=torch.float64)
        self.n_slots = n_slots
        self.k = None if k is None else k.detach().to("cpu", torch.float64)
        self.encoder = encoder if encoder is not None else ScatteringEncoder(x, n_slots, k=k)
        self.decoder = decoder if decoder is not None else ScatteringDecoder(x, n_slots, k=k)
        self.forward_mode: Mode = forward_mode
        self.inverse_mode: Mode = inverse_mode

    def set_mode(self, forward: Mode | None = None, inverse: Mode | None = None) -> "ScatteringNeuralOperator":
        if forward is not None:
            self.forward_mode = forward
        if inverse is not None:
            self.inverse_mode = inverse
        return self

    # -- the three stages ---------------------------------------------------
    def encode(self, u0: Tensor) -> SolitonSlots:
        if self.forward_mode == "learned":
            return self.encoder(u0.to(_param_dtype(self.encoder)).to(self._device))
        data = kdv_direct(u0.detach().to("cpu", torch.float64), self.x64, self.k, reflection=self.k is not None)
        return SolitonSlots.from_scattering(data, self.n_slots)

    @staticmethod
    def evolve(slots: SolitonSlots, t: Tensor) -> SolitonSlots:
        return slots.evolve(t)

    def decode(self, slots: SolitonSlots) -> Tensor:
        if self.inverse_mode == "learned":
            return self.decoder(slots.to(_param_dtype(self.decoder)).to(self._device))
        x64 = self.x64
        slots = slots.to("cpu")
        if slots.R is not None:
            # general data: Marchenko, one sample at a time (expensive)
            flat, lead = slots.flatten()
            d = flat.to(torch.float64).to_scattering()
            return torch.stack([kdv_marchenko(_index_data(d, i), x64)
                                for i in range(d.kappa.shape[0])]).reshape(*lead, -1)
        return kdv_reflectionless(slots.to(torch.float64).to_scattering(), x64)

    @property
    def _device(self) -> torch.device:
        return next(self.decoder.parameters()).device

    def forward(self, u0: Tensor, t: Tensor) -> Tensor:
        """u0: (B, n_x); t: (Nt,) -> (B, Nt, n_x) on the networks' device
        (float32 for learned decoding, float64 for exact)."""
        slots = self.encode(u0)
        t = torch.as_tensor(t).to(slots.kappa.device, slots.kappa.dtype)
        return self.decode(self.evolve(slots, t))


def _index_data(d: KdVScatteringData, i: int) -> KdVScatteringData:
    return KdVScatteringData(d.kappa[i], d.log_c[i], d.mask[i], d.k, None if d.R is None else d.R[i])


# ---------------------------------------------------------------------------
# FNO baseline
# ---------------------------------------------------------------------------

class _SpectralConv1d(nn.Module):
    def __init__(self, c: int, modes: int):
        super().__init__()
        self.modes = modes
        self.w = nn.Parameter(torch.randn(c, c, modes, dtype=complex_dtype(torch.get_default_dtype())) / (c * c))

    def forward(self, h: Tensor) -> Tensor:
        H = torch.fft.rfft(h)
        out = torch.zeros_like(H)
        m = min(self.modes, H.shape[-1])
        out[..., :m] = torch.einsum("bim,iom->bom", H[..., :m], self.w[..., :m])
        return torch.fft.irfft(out, n=h.shape[-1])


class FNO1dBaseline(nn.Module):
    """Standard Fourier neural operator  (u0, t) -> u(x,t).

    Input channels: u0, the coordinate, and t (broadcast).  Used to compare
    "learn in Fourier space" against "learn the transforms, evolve exactly in
    scattering space"."""

    def __init__(self, x: Tensor, width: int = 64, modes: int = 48, layers: int = 4, t_scale: float = 1.0):
        super().__init__()
        self.register_buffer("x", x.to(torch.get_default_dtype()))
        self.x_scale = float(x.abs().max())
        self.t_scale = t_scale
        self.lift = nn.Conv1d(3, width, 1)
        self.spec = nn.ModuleList([_SpectralConv1d(width, modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv1d(width, 128, 1), nn.GELU(), nn.Conv1d(128, 1, 1))

    def forward(self, u0: Tensor, t: Tensor) -> Tensor:
        B, n = u0.shape
        dt = _param_dtype(self)
        t = torch.as_tensor(t).to(dt).to(u0.device).reshape(-1)
        Nt = t.numel()
        u = u0.to(dt)[:, None, :].expand(B, Nt, n).reshape(B * Nt, n)
        tt = (t / self.t_scale)[None, :, None].expand(B, Nt, n).reshape(B * Nt, n)
        xc = (self.x / self.x_scale).to(dt).expand(B * Nt, n)
        h = self.lift(torch.stack([u / 2.0, xc, tt], 1))
        for sc, pw in zip(self.spec, self.pw):
            h = F.gelu(sc(h) + pw(h))
        return (2.0 * self.proj(h)[:, 0]).reshape(B, Nt, n)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

@dataclass
class KdVDataset:
    """Initial conditions with exact scattering data.

    u0: (S, n) ; slots: SolitonSlots with (S, N) fields (presence 0/1)."""

    x: Tensor
    u0: Tensor
    slots: SolitonSlots

    def __len__(self) -> int:
        return self.u0.shape[0]

    def batch(self, idx: Tensor) -> tuple[Tensor, SolitonSlots]:
        return self.u0[idx], self.slots[idx]

    def to(self, device) -> "KdVDataset":
        return KdVDataset(self.x.to(device), self.u0.to(device), self.slots.to(device))


def exact_kdv_solution(slots: SolitonSlots, x: Tensor, t: Tensor) -> Tensor:
    """Ground-truth u(x,t) for reflectionless slot data (closed form, float64).

    slots fields (B, N); t (Nt,) -> (B, Nt, n)."""
    with torch.no_grad():
        s = slots.to(torch.float64)
        data = s.to_scattering(0.5)
        return kdv_reflectionless(data.evolve(torch.as_tensor(t, dtype=torch.float64, device=x.device)), x.double())


def generate_kdv_soliton_data(n_samples: int, x: Tensor, n_slots: int = 4,
                              n_solitons: tuple[int, int] = (1, 3),
                              kappa_range: tuple[float, float] = (0.4, 1.2),
                              x0_range: tuple[float, float] = (-12.0, 4.0),
                              min_kappa_gap: float = 0.05, seed: int = 0,
                              t_check: tuple[float, ...] = (0.0,), edge_tol: float = 1e-4,
                              edge_frac: float = 0.04, device=None) -> KdVDataset:
    """Random multi-soliton initial conditions.

    Scattering data are sampled directly (kappa uniform, x0 uniform, number of
    solitons uniform in n_solitons), and u(x,0) is produced by the exact
    inverse transform -- so every pair (u0, scattering data) is exact.
    Distinct kappas are kept min_kappa_gap apart (equal kappas would be a
    degenerate, non-generic case).

    Samples are rejected and redrawn unless |u| < edge_tol on the outer
    edge_frac of the grid at every time in t_check: interaction phase shifts
    can push a soliton several widths away from its nominal x0, and a
    profile truncated by the domain no longer has the sampled scattering data."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = n_solitons
    N = n_slots
    n_edge = max(2, int(edge_frac * x.numel()))
    t_check_t = torch.tensor(t_check, dtype=torch.float64)
    kept: list[tuple[Tensor, Tensor, Tensor]] = []
    while len(kept) < n_samples:
        S = max(16, 2 * (n_samples - len(kept)))
        kappa = torch.ones(S, N, dtype=torch.float64)
        x0 = torch.zeros(S, N, dtype=torch.float64)
        pres = torch.zeros(S, N, dtype=torch.float64)
        counts = torch.randint(lo, hi + 1, (S,), generator=g)
        for i in range(S):
            m = int(counts[i])
            while True:
                ks = kappa_range[0] + (kappa_range[1] - kappa_range[0]) * torch.rand(m, generator=g, dtype=torch.float64)
                ks, _ = ks.sort(descending=True)
                if m == 1 or (-(ks.diff())).min() > min_kappa_gap:
                    break
            kappa[i, :m] = ks
            x0[i, :m] = x0_range[0] + (x0_range[1] - x0_range[0]) * torch.rand(m, generator=g, dtype=torch.float64)
            pres[i, :m] = 1.0
        u = exact_kdv_solution(SolitonSlots(kappa, x0, pres), x, t_check_t)            # (S, Nt, n)
        edge = torch.cat([u[..., :n_edge], u[..., -n_edge:]], -1).abs().amax((-1, -2))
        for i in torch.nonzero(edge < edge_tol).flatten().tolist():
            kept.append((kappa[i], x0[i], pres[i]))
    kappa, x0, pres = (torch.stack(v)[:n_samples] for v in zip(*kept))
    slots = SolitonSlots(kappa, x0, pres)
    u0 = exact_kdv_solution(slots, x, torch.zeros(1))[:, 0]
    ds = KdVDataset(x.double(), u0, slots)
    return ds.to(device) if device is not None else ds


def generate_kdv_general_data(n_samples: int, x: Tensor, n_slots: int = 4, k: Tensor | None = None,
                              n_bumps: tuple[int, int] = (1, 3), depth_range: tuple[float, float] = (0.5, 3.0),
                              width_range: tuple[float, float] = (0.5, 2.0), positive_prob: float = 0.2,
                              center_range: tuple[float, float] = (-8.0, 4.0), seed: int = 0) -> KdVDataset:
    """Random non-reflectionless initial conditions (sums of Gaussian wells and
    bumps) labelled by the *exact direct transform* (slots + R on k).

    This is the 'random initial conditions -> exact IST -> pairs' path for
    data with radiation.  It runs kdv_direct per sample, so it is slower."""
    g = torch.Generator().manual_seed(seed)
    S = n_samples
    xd = x.double()
    u0 = torch.zeros(S, xd.numel(), dtype=torch.float64)
    for i in range(S):
        m = int(torch.randint(n_bumps[0], n_bumps[1] + 1, (1,), generator=g))
        for _ in range(m):
            A = depth_range[0] + (depth_range[1] - depth_range[0]) * torch.rand(1, generator=g, dtype=torch.float64)
            if torch.rand(1, generator=g) < positive_prob:
                A = -0.5 * A
            w = width_range[0] + (width_range[1] - width_range[0]) * torch.rand(1, generator=g, dtype=torch.float64)
            c = center_range[0] + (center_range[1] - center_range[0]) * torch.rand(1, generator=g, dtype=torch.float64)
            u0[i] -= A * torch.exp(-((xd - c) / w) ** 2)
    data = kdv_direct(u0, xd, k, reflection=k is not None)
    return KdVDataset(xd, u0, SolitonSlots.from_scattering(data, n_slots))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def potential_loss(pred: Tensor, target: Tensor, relative: bool = True) -> Tensor:
    """(Relative) L2 error over the last axis, averaged over everything else."""
    err = (pred - target.to(pred.dtype)).norm(dim=-1)
    if relative:
        err = err / target.to(pred.dtype).norm(dim=-1).clamp_min(1e-6)
    return err.mean()


def scattering_loss(pred: SolitonSlots, target: SolitonSlots, x_scale: float = 1.0,
                    w_presence: float = 1.0, w_kappa: float = 1.0, w_x0: float = 1.0,
                    w_R: float = 1.0) -> Tensor:
    """Presence BCE + presence-masked MSE on kappa and x0 (scaled) + MSE on R."""
    m = target.presence.to(pred.kappa.dtype)
    loss = w_presence * F.binary_cross_entropy_with_logits(pred.logits, m)
    denom = m.sum().clamp_min(1.0)
    loss = loss + w_kappa * (m * (pred.kappa - target.kappa.to(m.dtype)) ** 2).sum() / denom
    loss = loss + w_x0 * (m * ((pred.x0 - target.x0.to(m.dtype)) / x_scale) ** 2).sum() / denom
    if pred.R is not None and target.R is not None:
        loss = loss + w_R * (pred.R - target.R.to(pred.R.dtype)).abs().pow(2).mean()
    return loss


def _d1(u: Tensor, h: float) -> Tensor:
    """4th-order central first derivative (zero-padded edges)."""
    # F.conv1d is a cross-correlation: weights are in u[i-2], ..., u[i+2] order
    k = torch.tensor([1, -8, 0, 8, -1], dtype=u.dtype, device=u.device) / (12 * h)
    return F.conv1d(u.reshape(-1, 1, u.shape[-1]), k[None, None], padding=2).reshape(u.shape)


def _d3(u: Tensor, h: float) -> Tensor:
    """4th-order central third derivative."""
    k = torch.tensor([1, -8, 13, 0, -13, 8, -1], dtype=u.dtype, device=u.device) / (8 * h ** 3)
    return F.conv1d(u.reshape(-1, 1, u.shape[-1]), k[None, None], padding=3).reshape(u.shape)


def kdv_residual(u_fn: Callable[[Tensor], Tensor], t: Tensor, h: float, dt: float = 1e-2) -> Tensor:
    """Pointwise residual of u_t - 6 u u_x + u_xxx at times t.

    u_fn(t) -> (..., Nt, n).  u_t by a 4th-order central difference in t
    (exact evolution makes u smooth in t), x-derivatives by 4th-order stencils.
    Edges (3 points each side) are dropped."""
    t = torch.as_tensor(t)
    u = u_fn(t)
    ut = (-u_fn(t + 2 * dt) + 8 * u_fn(t + dt) - 8 * u_fn(t - dt) + u_fn(t - 2 * dt)) / (12 * dt)
    r = ut - 6 * u * _d1(u, h) + _d3(u, h)
    return r[..., 3:-3]


def kdv_residual_loss(u_fn: Callable[[Tensor], Tensor], t: Tensor, h: float, dt: float = 1e-2) -> Tensor:
    """Physics-informed loss: mean squared KdV residual, normalised by the
    mean square of u_xxx (so it is scale-free and ~1 for a random field)."""
    u = u_fn(torch.as_tensor(t))
    scale = _d3(u, h)[..., 3:-3].pow(2).mean().detach().clamp_min(1e-8)
    return kdv_residual(u_fn, t, h, dt).pow(2).mean() / scale


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _batches(n: int, batch_size: int, generator: torch.Generator) -> Iterator[Tensor]:
    while True:
        perm = torch.randperm(n, generator=generator)
        for i in range(0, n - batch_size + 1, batch_size):
            yield perm[i:i + batch_size]


def train(op: ScatteringNeuralOperator, data: KdVDataset, stage: Literal["encoder", "decoder", "end_to_end"],
          steps: int = 2000, batch_size: int = 64, lr: float = 2e-3, t_max: float = 2.0, n_times: int = 4,
          physics_weight: float = 0.0, scattering_weight: float = 0.1, log_every: int = 200,
          seed: int = 0, verbose: bool = True) -> list[dict]:
    """Train one stage of the neural IST.

    encoder     : encoder only, on the scattering loss against exact data.
    decoder     : decoder only, teacher-forced with the exact scattering data
                  evolved to random t in [0, t_max]; loss vs exact u(x,t).
    end_to_end  : learned encoder -> exact evolution -> learned decoder, loss
                  vs exact u(x,t), plus scattering_weight * scattering loss
                  (keeps the latent space *physical*) and, optionally,
                  physics_weight * KdV residual.
    Targets for u(x,t) come from the exact closed-form inverse on the fly.
    Returns a list of logged metrics.
    """
    params = {"encoder": op.encoder.parameters(), "decoder": op.decoder.parameters(),
              "end_to_end": op.parameters()}[stage]
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed)
    batches = _batches(len(data), batch_size, g)
    h = float(data.x[1] - data.x[0])
    dev = op._device                    # networks' device; data and exact targets stay on CPU
    ndt = _param_dtype(op)
    op.train()
    hist: list[dict] = []
    t0 = time.time()
    run: dict[str, float] = {}
    for step in range(1, steps + 1):
        idx = next(batches)
        u0, target = data.batch(idx)
        target_d = target.to(ndt).to(dev)
        u0 = u0.to(ndt).to(dev)
        logs: dict[str, float] = {}
        if stage == "encoder":
            loss = scattering_loss(op.encoder(u0), target_d)
            logs["scatter"] = loss.item()
        else:
            t64 = t_max * torch.rand(n_times, generator=g, dtype=torch.float64)
            u_true = exact_kdv_solution(target, data.x, t64).to(ndt).to(dev)
            t = t64.to(ndt).to(dev)
            if stage == "decoder":
                pred = op.decoder(target_d.evolve(t))
                loss = potential_loss(pred, u_true)
                logs["u_rel"] = loss.item()
            else:
                slots = op.encoder(u0)
                pred = op.decoder(slots.evolve(t))
                lu = potential_loss(pred, u_true)
                ls = scattering_loss(slots, target_d)
                loss = lu + scattering_weight * ls
                logs["u_rel"], logs["scatter"] = lu.item(), ls.item()
                if physics_weight > 0:
                    sub = slots[:8]   # the residual needs 5 decoder passes: use a sub-batch
                    lp = kdv_residual_loss(lambda tt: op.decoder(sub.evolve(tt)), t, h)
                    loss = loss + physics_weight * lp
                    logs["physics"] = lp.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        for kk, v in logs.items():
            run[kk] = 0.9 * run.get(kk, v) + 0.1 * v
        if step % log_every == 0 or step == steps:
            rec = {"step": step, "time": time.time() - t0, **run}
            hist.append(rec)
            if verbose:
                print(f"  [{stage}] step {step:5d}  " + "  ".join(f"{a}={b:.4g}" for a, b in run.items())
                      + f"  ({rec['time']:.0f}s)")
    op.eval()
    return hist


def train_baseline(model: FNO1dBaseline, data: KdVDataset, steps: int = 2000, batch_size: int = 64,
                   lr: float = 2e-3, t_max: float = 2.0, n_times: int = 4, log_every: int = 200,
                   seed: int = 0, verbose: bool = True) -> list[dict]:
    """Train the FNO baseline on exactly the same (u0, t, u(x,t)) distribution."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed)
    batches = _batches(len(data), batch_size, g)
    dev = next(model.parameters()).device
    ndt = _param_dtype(model)
    model.train()
    hist: list[dict] = []
    t0 = time.time()
    run = None
    for step in range(1, steps + 1):
        idx = next(batches)
        u0, target = data.batch(idx)
        t64 = t_max * torch.rand(n_times, generator=g, dtype=torch.float64)
        u_true = exact_kdv_solution(target, data.x, t64).to(ndt).to(dev)
        loss = potential_loss(model(u0.to(ndt).to(dev), t64.to(ndt).to(dev)), u_true)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = loss.item() if run is None else 0.9 * run + 0.1 * loss.item()
        if step % log_every == 0 or step == steps:
            hist.append({"step": step, "time": time.time() - t0, "u_rel": run})
            if verbose:
                print(f"  [fno] step {step:5d}  u_rel={run:.4g}  ({time.time() - t0:.0f}s)")
    model.eval()
    return hist


__all__ = [
    "SolitonSlots", "ScatteringEncoder", "ScatteringDecoder", "ScatteringNeuralOperator", "FNO1dBaseline",
    "KdVDataset", "generate_kdv_soliton_data", "generate_kdv_general_data", "exact_kdv_solution",
    "potential_loss", "scattering_loss", "kdv_residual", "kdv_residual_loss", "train", "train_baseline",
]
