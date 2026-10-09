"""Tests for torch_ist.neural (CPU, small sizes, fast)."""

import pytest
import torch

import torch_ist as ti
from torch_ist.neural import (
    FNO1dBaseline,
    ScatteringDecoder,
    ScatteringNeuralOperator,
    SolitonSlots,
    exact_kdv_solution,
    generate_kdv_soliton_data,
    kdv_residual_loss,
    potential_loss,
    scattering_loss,
    train,
)

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def x():
    return torch.linspace(-20, 30, 256)


@pytest.fixture(scope="module")
def ds(x):
    return generate_kdv_soliton_data(48, x, n_slots=3, n_solitons=(1, 3), x0_range=(-10.0, 0.0),
                                     t_check=(0.0, 2.0), seed=0)


def test_slots_roundtrip_and_evolution(x):
    data = ti.KdVScatteringData.solitons([1.1, 0.6], [-4.0, 3.0])
    slots = SolitonSlots.from_scattering(data, 4)
    assert slots.presence.tolist() == [1, 1, 0, 0]
    assert torch.allclose(slots.x0[:2], torch.tensor([-4.0, 3.0]))
    t = torch.tensor([0.0, 0.7, 1.5])
    # slot evolution (x0 += 4 kappa^2 t) == norming-constant evolution
    u_slots = ti.kdv_reflectionless(slots.evolve(t).to_scattering(), x)
    u_data = ti.kdv_reflectionless(data.evolve(t), x)
    assert (u_slots - u_data).abs().max() < 1e-12


def test_generator_is_exact_and_in_domain(x, ds):
    assert ds.u0[:, :10].abs().max() < 1e-4 and ds.u0[:, -10:].abs().max() < 1e-4
    d = ti.kdv_direct(ds.u0[:8], x, reflection=False)
    rec = SolitonSlots.from_scattering(d, 3)
    assert torch.equal(rec.presence, ds.slots.presence[:8])
    m = rec.presence.bool()
    assert (rec.kappa - ds.slots.kappa[:8])[m].abs().max() < 1e-4
    assert (rec.x0 - ds.slots.x0[:8])[m].abs().max() < 1e-3


def test_exact_exact_operator_is_classical_ist(x, ds):
    op = ScatteringNeuralOperator(x, 3).set_mode("exact", "exact")
    t = torch.tensor([0.0, 1.0, 2.0])
    u = op(ds.u0[:6], t)
    assert u.shape == (6, 3, x.numel())
    assert (u - exact_kdv_solution(ds.slots[:6], x, t)).abs().max() < 5e-3


@pytest.mark.parametrize("fm,im", [("exact", "learned"), ("learned", "exact"), ("learned", "learned")])
def test_modes_shapes_and_gradients(x, ds, fm, im):
    op = ScatteringNeuralOperator(x, 3).set_mode(fm, im)
    with torch.no_grad():   # untrained presence logits may all be < 0 -> exact decoder sees no solitons
        op.encoder.mlp[-1].bias[:3] = 5.0
    t = torch.tensor([0.0, 0.5])
    u = op(ds.u0[:4], t)
    assert u.shape == (4, 2, x.numel())
    target = exact_kdv_solution(ds.slots[:4], x, t)
    potential_loss(u.double(), target).backward()
    if fm == "learned":
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in op.encoder.parameters())
    if im == "learned":
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in op.decoder.parameters())


def test_decoder_translation_equivariance():
    x = torch.linspace(-60, 60, 1201)   # receptive field is ~250 points each side
    dec = ScatteringDecoder(x, 2)
    with torch.no_grad():
        for p in dec.proj.parameters():           # make the interaction CNN non-trivial
            p.normal_(0, 0.1)
    h = float(x[1] - x[0])
    s = SolitonSlots(torch.tensor([[1.0, 0.6]]), torch.tensor([[-3.0, 2.0]]), torch.tensor([[1.0, 1.0]]))
    shift = 10
    s2 = SolitonSlots(s.kappa, s.x0 + shift * h, s.presence)
    with torch.no_grad():
        u1, u2 = dec(s), dec(s2)
    # equivariant away from the zero-padded boundaries (receptive field ~ 250 points)
    assert (u2[0, 300 + shift:-300] - u1[0, 300:-300 - shift]).abs().max() < 1e-6


def test_physics_residual(x):
    h = float(x[1] - x[0])
    t = torch.linspace(0.2, 1.0, 4)
    exact = kdv_residual_loss(lambda tt: ti.kdv_soliton(x, tt, 1.0), t, h)
    wrong = kdv_residual_loss(lambda tt: ti.kdv_soliton(x, tt, 1.0) * 1.2, t, h)  # not a solution
    assert exact < 1e-3 and wrong > 1e-2


def test_scattering_loss_zero_at_target(ds):
    tgt = ds.slots[:5]
    logits = torch.where(tgt.presence > 0, 30.0, -30.0)
    pred = SolitonSlots(tgt.kappa, tgt.x0, tgt.presence, logits)
    assert scattering_loss(pred, tgt) < 1e-10


def test_training_reduces_loss(x, ds):
    torch.manual_seed(0)
    op = ScatteringNeuralOperator(x, 3)
    hist = train(op, ds, "encoder", steps=40, batch_size=16, lr=2e-3, log_every=10, verbose=False)
    assert hist[-1]["scatter"] < hist[0]["scatter"]
    hist = train(op, ds, "decoder", steps=30, batch_size=8, n_times=2, log_every=10, verbose=False)
    assert hist[-1]["u_rel"] < hist[0]["u_rel"]


def test_fno_baseline_shape(x, ds):
    fno = FNO1dBaseline(x, width=16, modes=16)
    assert fno(ds.u0[:3], torch.tensor([0.0, 1.0])).shape == (3, 2, x.numel())
