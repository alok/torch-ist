"""Matplotlib helpers: spacetime heatmaps, IST-vs-PDE comparisons, spectra."""

from __future__ import annotations

from typing import Sequence

import matplotlib


import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from .direct import KdVScatteringData, NLSScatteringData  # noqa: E402


def _np(a) -> np.ndarray:
    if isinstance(a, Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def spacetime_heatmap(field, x, t, ax=None, title: str = "", cmap: str = "magma",
                      label: str = "", symmetric: bool = False):
    """Heatmap of field(t, x) with t on the vertical axis.  field: (Nt, Nx).
    Complex fields are shown as |field|."""
    f = _np(field)
    if np.iscomplexobj(f):
        f = np.abs(f)
    x, t = _np(x), _np(t)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    kw = {}
    if symmetric:
        m = np.abs(f).max()
        kw = dict(vmin=-m, vmax=m)
    im = ax.imshow(f, origin="lower", aspect="auto", cmap=cmap,
                   extent=[x[0], x[-1], t[0], t[-1]], interpolation="nearest", **kw)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, label=label)
    return ax


def plot_profiles(x, fields: dict, ax=None, title: str = ""):
    """Overlay several 1-D profiles {label: values}."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    x = _np(x)
    for i, (lab, f) in enumerate(fields.items()):
        f = _np(f)
        if np.iscomplexobj(f):
            f = np.abs(f)
        ax.plot(x, f, label=lab, lw=2.2 if i == 0 else 1.4, ls="-" if i == 0 else "--")
    ax.legend()
    ax.set_xlabel("x")
    if title:
        ax.set_title(title)
    return ax


def compare_solutions(x, t, ist, pde, title: str = "", value_label: str = "u",
                      cmap: str = "magma", path: str | None = None, symmetric: bool = False):
    """Three panels: IST, reference PDE, and |difference|."""
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    spacetime_heatmap(ist, x, t, axs[0], "IST", cmap, value_label, symmetric)
    spacetime_heatmap(pde, x, t, axs[1], "direct PDE integration", cmap, value_label, symmetric)
    diff = np.abs(_np(ist) - _np(pde))
    spacetime_heatmap(diff, x, t, axs[2], f"|IST - PDE|  (max {diff.max():.2e})", "viridis", "error")
    if title:
        fig.suptitle(title)
    if path:
        fig.savefig(path, dpi=130)
    return fig


def plot_kdv_spectrum(data: KdVScatteringData, ax=None):
    """Bound states and |R(k)| for (unbatched) KdV scattering data."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    if data.R is not None:
        ax.plot(_np(data.k), np.abs(_np(data.R)), label="|R(k)|")
    kap = _np(data.kappa[data.mask])
    ax.scatter(np.zeros_like(kap), kap, color="C3", zorder=3, label=r"bound states $k=i\kappa_n$ (plotted at $\kappa_n$)")
    ax.set_xlabel("k")
    ax.legend()
    return ax


def plot_nls_spectrum(data: NLSScatteringData, ax=None):
    """Eigenvalues in the complex zeta-plane and |r(xi)| on the real axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    z = _np(data.zeta[data.mask])
    ax.scatter(z.real, z.imag, color="C3", zorder=3, label=r"eigenvalues $\zeta_n$")
    if data.r is not None:
        ax.plot(_np(data.xi), np.abs(_np(data.r)), label=r"$|r(\xi)|$")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"Re $\zeta$")
    ax.set_ylabel(r"Im $\zeta$")
    ax.legend()
    return ax


def track_peaks(field, x, threshold: float = 0.5) -> list[np.ndarray]:
    """Positions of local maxima of |field| above threshold*max, per time row."""
    f = np.abs(_np(field))
    x = _np(x)
    out = []
    for row in f:
        m = row.max()
        idx = np.where((row[1:-1] > row[:-2]) & (row[1:-1] >= row[2:]) & (row[1:-1] > threshold * m))[0] + 1
        out.append(x[idx])
    return out


__all__: Sequence[str] = [
    "spacetime_heatmap", "plot_profiles", "compare_solutions",
    "plot_kdv_spectrum", "plot_nls_spectrum", "track_peaks",
]
