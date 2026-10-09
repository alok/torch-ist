"""Tests for torch-ist.

Groups:
  * direct scattering against closed-form spectra (Poschl-Teller, Satsuma-Yajima)
  * forward -> inverse round trips (reflectionless and general Marchenko)
  * known exact solutions (solitons, phase shifts)
  * IST vs direct PDE integration
  * batching, devices, autodiff
"""

import math

import pytest
import torch

import torch_ist as ti
from torch_ist.utils import sech

torch.set_default_dtype(torch.float64)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def x():
    return torch.linspace(-20, 20, 801)


@pytest.fixture(scope="module")
def xs():
    """Coarser grid for the O(n^4) Marchenko solver."""
    return torch.linspace(-15, 15, 301)


# ---------------------------------------------------------------------------
# Direct scattering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nu", [1, 2, 3])
def test_poschl_teller_bound_states(x, nu):
    """u = -nu(nu+1) sech^2 x: kappa_j = nu - j, reflectionless."""
    d = ti.kdv_direct(-nu * (nu + 1) * sech(x) ** 2, x)
    expected = torch.arange(nu, 0, -1, dtype=torch.float64)
    assert torch.allclose(d.kappa[d.mask], expected, atol=1e-6)
    assert d.R.abs().max() < 1e-3


def test_poschl_teller_noninteger(x):
    """u = -3 sech^2: nu = (sqrt(13)-1)/2, kappa = nu, nu-1, and R != 0."""
    nu = (math.sqrt(13) - 1) / 2
    d = ti.kdv_direct(-3 * sech(x) ** 2, x)
    assert torch.allclose(d.kappa, torch.tensor([nu, nu - 1]), atol=1e-6)
    assert d.R.abs().max() > 0.5


def test_single_soliton_norming_constant(x):
    """-2 sech^2(x - x0): c^2 = 2 e^{2 x0}."""
    d = ti.kdv_direct(-2 * sech(x - 1.5) ** 2, x, reflection=False)
    assert torch.allclose(d.c2, torch.tensor([2 * math.exp(3.0)]), rtol=1e-5)


def test_unitarity(x):
    """|R|^2 + |T|^2 = 1 for real k (T = 1/a)."""
    k = torch.linspace(0.05, 5, 50)
    a, b = ti.schrodinger_ab(-1.7 * torch.exp(-x ** 2) + 0.3 * sech(x - 2), k.to(torch.complex128),
                             ti.Grid.from_x(x))
    assert torch.allclose((b / a).abs() ** 2 + (1 / a).abs() ** 2, torch.ones(50), atol=1e-10)


@pytest.mark.parametrize("A", [1.0, 2.0, 2.7])
def test_zs_sech_eigenvalues(x, A):
    """q = A sech x: zeta_j = i (A - 1/2 - j) for A - 1/2 - j > 0."""
    d = ti.nls_direct(A * sech(x), x, reflection=False)
    expected = [A - 0.5 - j for j in range(int(math.ceil(A - 0.5)))]
    assert torch.allclose(d.zeta.imag, torch.tensor(expected), atol=1e-6)
    assert d.zeta.real.abs().max() < 1e-6


def test_zs_moving_soliton_data(x):
    q0 = ti.nls_soliton(x, 0.0, eta=0.6, xi=0.3, x0=1.0, phase=0.4)
    d = ti.nls_direct(q0, x, reflection=False)
    ref = ti.NLSScatteringData.solitons(0.6, 0.3, 1.0, 0.4)
    assert torch.allclose(d.zeta, ref.zeta, atol=1e-6)
    assert torch.allclose(torch.exp(d.log_C), torch.exp(ref.log_C), rtol=1e-4)


def test_separated_two_soliton_norming(x):
    """Each eigenfunction must be matched near its own peak (tunnelling)."""
    u0 = ti.kdv_from_solitons(x, torch.tensor([0.0]), [1.2, 0.7], [-12.0, 4.0])[0]
    d = ti.kdv_direct(u0, x, reflection=False)
    ref = ti.KdVScatteringData.solitons([1.2, 0.7], [-12.0, 4.0])
    assert torch.allclose(d.log_c, ref.log_c, atol=1e-4)


# ---------------------------------------------------------------------------
# Forward -> inverse round trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nu", [1, 2, 3])
def test_roundtrip_kdv_reflectionless(x, nu):
    u0 = -nu * (nu + 1) * sech(x) ** 2
    u = ti.kdv_inverse(ti.kdv_direct(u0, x, reflection=False), x)
    assert (u - u0).abs().max() < 1e-3 * nu * (nu + 1)


@pytest.mark.parametrize("A", [1.0, 2.0, 3.0])
def test_roundtrip_nls_reflectionless(x, A):
    q0 = (A * sech(x)).to(torch.complex128)
    q = ti.nls_inverse(ti.nls_direct(q0, x, reflection=False), x)
    assert (q - q0).abs().max() < 1e-3


@pytest.mark.parametrize("u0fn,tol", [
    (lambda z: -3 * sech(z) ** 2, 1e-2),          # 2 solitons + radiation
    (lambda z: -2.5 * torch.exp(-z ** 2), 1e-2),  # 1 soliton + radiation
    (lambda z: 1.0 * torch.exp(-z ** 2), 5e-3),   # repulsive: radiation only
    (lambda z: -1.5 * sech(z + 3) ** 2 - torch.exp(-(z - 4) ** 2), 1e-2),  # asymmetric
])
def test_roundtrip_kdv_marchenko(xs, u0fn, tol):
    u0 = u0fn(xs)
    u = ti.kdv_marchenko(ti.kdv_direct(u0, xs), xs)
    assert (u - u0).abs().max() < tol


@pytest.mark.parametrize("q0fn,tol", [
    (lambda z: 1.2 * sech(z), 2e-3),
    (lambda z: 1.5 * sech(z) * torch.exp(0.5j * z + 0.3j * torch.tanh(z)), 1e-2),  # chirped
    (lambda z: 0.4 * torch.exp(-z ** 2), 2e-3),   # radiation only
])
def test_roundtrip_nls_marchenko(xs, q0fn, tol):
    q0 = q0fn(xs).to(torch.complex128)
    q = ti.nls_marchenko(ti.nls_direct(q0, xs), xs)
    assert (q - q0).abs().max() < tol


def test_marchenko_matches_closed_form_for_solitons(xs):
    """With R = 0 the general solver must reproduce the closed form, even for
    widely separated (ill-conditioned) solitons."""
    data = ti.KdVScatteringData.solitons([1.5, 1.0, 0.6], [-9.0, 0.0, 8.0])
    assert (ti.kdv_marchenko(data, xs) - ti.kdv_reflectionless(data, xs)).abs().max() < 5e-3
    ndata = ti.NLSScatteringData.solitons([0.8, 0.5], [0.2, -0.3], [-7.0, 6.0], [0.0, 1.0])
    assert (ti.nls_marchenko(ndata, xs) - ti.nls_reflectionless(ndata, xs)).abs().max() < 5e-3


# ---------------------------------------------------------------------------
# Known solutions
# ---------------------------------------------------------------------------

def test_kdv_single_soliton_propagation(x):
    t = torch.linspace(0, 3, 7)
    u, _ = ti.kdv_ist_solve(-2 * sech(x) ** 2, x, t, reflection=False)
    assert (u - ti.kdv_soliton(x, t, 1.0)).abs().max() < 1e-4


@pytest.mark.parametrize("kappa,x0", [(0.7, -3.0), (1.6, 2.0)])
def test_kdv_from_solitons_single(x, kappa, x0):
    t = torch.linspace(0, 1, 5)
    u = ti.kdv_from_solitons(x, t, [kappa], [x0])
    assert (u - ti.kdv_soliton(x, t, kappa, x0)).abs().max() < 1e-12


def test_kdv_two_soliton_phase_shift():
    """Asymptotic positions after a collision shift by log((k1+k2)/(k1-k2))/k."""
    k1, k2 = 1.2, 0.7
    x = torch.linspace(-60, 120, 3601)
    data = ti.KdVScatteringData.solitons([k1, k2], [0.0, 0.0])
    L = math.log((k1 + k2) / (k1 - k2))
    # Right-Marchenko normalisation: the rightmost soliton sits at its free
    # position.  t = +8: fast is rightmost, slow lags by L/k2.
    # t = -8: slow is rightmost, fast lags by L/k1.  Net shifts: +L/k1, -L/k2.
    expected_lag = {(8.0, k1): 0.0, (8.0, k2): -L / k2, (-8.0, k1): -L / k1, (-8.0, k2): 0.0}
    for tt in (-8.0, 8.0):
        u = ti.kdv_reflectionless(data.evolve(tt), x)
        for k in (k1, k2):
            centre = 4 * k ** 2 * tt + expected_lag[(tt, k)]
            window = (x - centre).abs() < 3 / k
            pos = x[window][u[window].argmin()].item()
            assert abs(pos - centre) < 0.03


def test_nls_soliton_closed_form(x):
    t = torch.linspace(0, 2, 5)
    q = ti.nls_from_solitons(x, t, [0.7], [0.25], [1.0], [0.3])
    assert (q - ti.nls_soliton(x, t, 0.7, 0.25, 1.0, 0.3)).abs().max() < 1e-12


def test_nls_breather_period(x):
    """2 sech x is periodic in |q| with period pi/2 (Satsuma-Yajima)."""
    q = ti.nls_from_solitons(x, torch.tensor([0.3, 0.3 + math.pi / 2]), [0.5, 1.5], 0.0, 0.0, [0.0, 0.0])
    assert (q[0].abs() - q[1].abs()).abs().max() < 1e-10


# ---------------------------------------------------------------------------
# IST vs direct PDE integration
# ---------------------------------------------------------------------------

def test_kdv_ist_vs_spectral_two_soliton():
    x = torch.linspace(-30, 40, 1024)
    t = torch.linspace(0, 4, 9)
    u0 = ti.kdv_from_solitons(x, torch.tensor([0.0]), [1.3, 0.8], [-10.0, 0.0])[0]
    u_ist, _ = ti.kdv_ist_solve(u0, x, t, reflection=False)
    u_pde = ti.kdv_spectral_solve(u0, x, t, dt=5e-4)
    assert (u_ist - u_pde).abs().max() < 1e-3


def test_kdv_ist_vs_spectral_radiation():
    x = torch.linspace(-20, 20, 241)
    t = torch.tensor([0.0, 0.4, 0.8])
    u0 = -2.5 * torch.exp(-x ** 2)
    u_ist, _ = ti.kdv_ist_solve(u0, x, t)
    h = (x[1] - x[0]).item()
    pad = 480  # periodic reference: keep fast radiation from wrapping around
    xp = x[0] - pad * h + h * torch.arange(len(x) + 2 * pad)
    u_pde = ti.kdv_spectral_solve(-2.5 * torch.exp(-xp ** 2), xp, t, dt=2e-4)[..., pad:pad + len(x)]
    assert (u_ist - u_pde).abs().max() < 1e-2


def test_nls_ist_vs_split_step_breather(x):
    t = torch.linspace(0, 1.5, 7)
    q0 = (2 * sech(x)).to(torch.complex128)
    q_ist, _ = ti.nls_ist_solve(q0, x, t, reflection=False)
    q_pde = ti.nls_split_step_solve(q0, x, t, dt=2e-4)
    assert (q_ist - q_pde).abs().max() < 1e-3


def test_nls_ist_vs_split_step_radiation():
    x = torch.linspace(-16, 16, 193)
    t = torch.tensor([0.0, 0.5, 1.0])
    q0 = (1.2 * sech(x)).to(torch.complex128)
    q_ist, _ = ti.nls_ist_solve(q0, x, t)
    q_pde = ti.nls_split_step_solve(q0, x, t, dt=5e-4)
    assert (q_ist - q_pde).abs().max() < 1e-2


# ---------------------------------------------------------------------------
# Batching, devices, autodiff
# ---------------------------------------------------------------------------

def test_batched_direct_matches_individual(x):
    u = torch.stack([-2 * sech(x) ** 2, -6 * sech(x - 1) ** 2, 0.5 * torch.exp(-x ** 2)])
    batch = ti.kdv_direct(u, x)
    assert batch.kappa.shape == (3, 2)
    assert batch.num_solitons.tolist() == [1, 2, 0]
    for i in range(3):
        single = ti.kdv_direct(u[i], x)
        n = single.kappa.shape[-1]
        assert torch.allclose(batch.kappa[i, :n], single.kappa)
        assert torch.allclose(batch.R[i], single.R)
    # batched inversion of padded data
    rec = ti.kdv_reflectionless(batch.reflectionless(), x)
    assert (rec[0] - u[0]).abs().max() < 1e-3
    assert (rec[1] - u[1]).abs().max() < 1e-2
    assert rec[2].abs().max() < 1e-12


def test_batched_nls(x):
    q = torch.stack([sech(x), 2 * sech(x)]).to(torch.complex128)
    d = ti.nls_direct(q, x, reflection=False)
    assert d.num_solitons.tolist() == [1, 2]
    rec = ti.nls_reflectionless(d, x)
    assert (rec - q).abs().max() < 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_device_roundtrip(device):
    x = torch.linspace(-15, 15, 301, device=device)
    u0 = -6 * sech(x) ** 2
    d = ti.kdv_direct(u0, x, reflection=False).to(device)
    u = ti.kdv_reflectionless(d.evolve(0.0), x)
    assert u.device.type == device
    assert (u - u0).abs().max() < 1e-2


def test_gradcheck_kdv_closed_form():
    x = torch.linspace(-5, 5, 21)
    kappa = torch.tensor([1.1, 0.6], requires_grad=True)
    log_c = torch.tensor([0.2, -0.1], requires_grad=True)

    def f(k, lc):
        return ti.kdv_reflectionless(ti.KdVScatteringData(k, lc, torch.ones(2, dtype=torch.bool)).evolve(0.3), x)

    assert torch.autograd.gradcheck(f, (kappa, log_c))


def test_gradcheck_nls_closed_form():
    x = torch.linspace(-5, 5, 15)
    eta = torch.tensor([0.6, 0.35], requires_grad=True)
    xi = torch.tensor([0.1, -0.2], requires_grad=True)
    x0 = torch.tensor([-1.0, 1.0], requires_grad=True)

    def f(e, v, p):
        return torch.view_as_real(ti.nls_from_solitons(x, torch.tensor([0.0, 0.5]), e, v, p, 0.0))

    assert torch.autograd.gradcheck(f, (eta, xi, x0))


def test_grad_through_direct_transform():
    """d kappa / dA for -A sech^2 x is 1/(2 sqrt(A + 1/4))."""
    x = torch.linspace(-15, 15, 401)
    A = torch.tensor(3.0, requires_grad=True)
    ti.kdv_direct(-A * sech(x) ** 2, x, reflection=False).kappa.max().backward()
    assert abs(A.grad.item() - 1 / (2 * math.sqrt(3.25))) < 1e-5


def test_grad_end_to_end_pipeline():
    """Gradient of a loss on u(x,t) w.r.t. the initial potential, through
    direct -> evolve -> closed-form inverse, against finite differences."""
    x = torch.linspace(-15, 15, 301)
    t = torch.tensor([0.5])
    w = torch.exp(-((x - 2) ** 2))

    def loss(A):
        u, _ = ti.kdv_ist_solve(-A * sech(x) ** 2, x, t, reflection=False)
        return (u[0] * w).sum()

    A = torch.tensor(2.0, requires_grad=True)
    loss(A).backward()
    fd = (loss(torch.tensor(2.0 + 1e-5)) - loss(torch.tensor(2.0 - 1e-5))) / 2e-5
    assert abs(A.grad.item() - fd.item()) < 1e-4 * max(1.0, abs(fd.item()))


def test_grad_through_marchenko(xs):
    """The general solver is differentiable too (w.r.t. the potential)."""
    A = torch.tensor(1.0, requires_grad=True)

    def loss(a):
        u0 = a * torch.exp(-xs ** 2)
        return ti.kdv_marchenko(ti.kdv_direct(u0, xs), xs)[150]

    loss(A).backward()
    fd = (loss(torch.tensor(1.0 + 1e-4)) - loss(torch.tensor(1.0 - 1e-4))) / 2e-4
    assert abs(A.grad.item() - fd.item()) < 1e-3
