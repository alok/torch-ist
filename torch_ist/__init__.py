"""torch-ist: the inverse scattering transform in PyTorch.

Direct scattering (Schrodinger / Zakharov-Shabat), Marchenko inversion with a
scaled low-rank soliton correction, closed-form N-soliton formulas, KdV and
focusing-NLS solvers, reference PDE integrators, plotting helpers, and a
neural operator that learns the transforms while evolving exactly in
scattering space (torch_ist.neural).
All core routines are batched, device-agnostic and differentiable.
"""

from .direct import (
    KdVScatteringData,
    NLSScatteringData,
    kdv_direct,
    nls_direct,
    schrodinger_ab,
    zakharov_shabat_ab,
)
from .inverse import (
    kdv_inverse,
    kdv_marchenko,
    kdv_reflectionless,
    nls_inverse,
    nls_marchenko,
    nls_reflectionless,
)
from .kdv import (
    kdv_evolve_and_invert,
    kdv_from_solitons,
    kdv_ist_solve,
    kdv_soliton,
    kdv_spectral_solve,
    kdv_two_soliton_phase_shifts,
)
from .nls import (
    nls_evolve_and_invert,
    nls_from_solitons,
    nls_ist_solve,
    nls_soliton,
    nls_split_step_solve,
)
from .utils import Grid, default_spectral_grid, graded_spectral_grid
from . import neural
from .neural import ScatteringNeuralOperator, SolitonSlots

__version__ = "0.1.0"

__all__ = [
    "Grid", "KdVScatteringData", "NLSScatteringData", "ScatteringNeuralOperator", "SolitonSlots", "neural",
    "default_spectral_grid", "graded_spectral_grid",
    "kdv_direct", "kdv_evolve_and_invert", "kdv_from_solitons", "kdv_inverse", "kdv_ist_solve",
    "kdv_marchenko", "kdv_reflectionless", "kdv_soliton", "kdv_spectral_solve",
    "kdv_two_soliton_phase_shifts",
    "nls_direct", "nls_evolve_and_invert", "nls_from_solitons", "nls_inverse", "nls_ist_solve",
    "nls_marchenko", "nls_reflectionless", "nls_soliton", "nls_split_step_solve",
    "schrodinger_ab", "zakharov_shabat_ab",
]
