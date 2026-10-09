"""Neural IST: a neural operator whose spectral domain is scattering space.

    u(x,0) --encoder--> {kappa_n, x0_n}  --exact evolution-->  {kappa_n, x0_n + 4 kappa_n^2 t}  --decoder--> u(x,t)

FNO learns in Fourier space, where linear PDEs diagonalise.  Here the network
learns only the direct and inverse scattering transforms; the time evolution
-- the part that is genuinely nonlinear in physical space -- is exact and free.

This script
  1. trains encoder, decoder, then end-to-end on random 1-3 soliton KdV data (t <= 2),
  2. trains an FNO baseline on the same data and steps,
  3. evaluates every exact/learned combination and the FNO on
       (a) held-out ICs, t in [0, 2]           (in distribution)
       (b) held-out ICs, t in (2, 4]           (time extrapolation)
       (c) 4-soliton ICs                        (unseen soliton count)
       (d) kappa in [1.2, 1.5]                  (unseen amplitudes, up to 4.5 vs 2.9)
  4. compares wall-clock time of exact vs neural IST,
  5. plots spacetime heatmaps.

Run with --quick for a fast smoke run (minutes on CPU).
"""

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from torch_ist.neural import (
    FNO1dBaseline,
    ScatteringNeuralOperator,
    exact_kdv_solution,
    generate_kdv_soliton_data,
    kdv_residual_loss,
    train,
    train_baseline,
)
from torch_ist.visualize import spacetime_heatmap
from _common import OUT

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
               ("cuda" if torch.cuda.is_available() else "cpu"))
args = p.parse_args()
dev = torch.device(args.device)
torch.manual_seed(0)

S = dict(n_train=600, enc=300, dec=300, e2e=200, fno=800) if args.quick else \
    dict(n_train=6000, enc=4000, dec=4000, e2e=2500, fno=10500)
N_SLOTS, T_TRAIN, T_EXT = 4, 2.0, 4.0
x = torch.linspace(-20, 30, 384, dtype=torch.float64)
h = float(x[1] - x[0])
print(f"device={dev}  grid n={x.numel()} h={h:.3f}  config={S}")

# ---------------------------------------------------------------------------
# data: scattering data sampled directly, u0 from the exact inverse transform
# ---------------------------------------------------------------------------
# every profile must stay inside the domain up to T_EXT (edges checked at t = 0 and T_EXT)
common = dict(n_slots=N_SLOTS, kappa_range=(0.4, 1.2), x0_range=(-12.0, 0.0), t_check=(0.0, T_EXT))
train_set = generate_kdv_soliton_data(S["n_train"], x, n_solitons=(1, 3), seed=1, **common)
tests = {
    "in-dist (t<=2)": (generate_kdv_soliton_data(256, x, n_solitons=(1, 3), seed=2, **common),
                       torch.linspace(0, T_TRAIN, 9)),
    "time extrap (2<t<=4)": (generate_kdv_soliton_data(256, x, n_solitons=(1, 3), seed=3, **common),
                             torch.linspace(T_TRAIN + 0.25, T_EXT, 8)),
    "4 solitons": (generate_kdv_soliton_data(256, x, n_solitons=(4, 4), seed=4, **common),
                   torch.linspace(0, T_TRAIN, 9)),
    "kappa in [1.2,1.5]": (generate_kdv_soliton_data(256, x, n_slots=N_SLOTS, n_solitons=(1, 2),
                                                     kappa_range=(1.2, 1.5), x0_range=(-12.0, -4.0),
                                                     t_check=(0.0, T_TRAIN), seed=5),
                           torch.linspace(0, T_TRAIN, 9)),
}

# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
op = ScatteringNeuralOperator(x.float(), N_SLOTS).to(dev)
n_enc = sum(q.numel() for q in op.encoder.parameters())
n_dec = sum(q.numel() for q in op.decoder.parameters())
t0 = time.time()
print(f"\n== training neural IST  (encoder {n_enc/1e3:.0f}k params, decoder {n_dec/1e3:.0f}k params)")
h_enc = train(op, train_set, "encoder", steps=S["enc"], lr=2e-3, log_every=max(1, S["enc"] // 5))
h_dec = train(op, train_set, "decoder", steps=S["dec"], lr=2e-3, t_max=T_TRAIN, log_every=max(1, S["dec"] // 5))
h_e2e = train(op, train_set, "end_to_end", steps=S["e2e"], lr=5e-4, t_max=T_TRAIN, physics_weight=1e-3,
              log_every=max(1, S["e2e"] // 5))
t_ours = time.time() - t0

fno = FNO1dBaseline(x.float(), width=64, modes=64, t_scale=T_TRAIN).to(dev)
n_fno = sum(q.numel() for q in fno.parameters())
print(f"\n== training FNO baseline ({n_fno/1e3:.0f}k params, {S['fno']} steps = same number of gradient steps)")
t0 = time.time()
h_fno = train_baseline(fno, train_set, steps=S["fno"], lr=2e-3, t_max=T_TRAIN, log_every=max(1, S["fno"] // 5))
t_fno = time.time() - t0
print(f"training time: neural IST {t_ours:.0f}s, FNO {t_fno:.0f}s")


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def rel_l2(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred, true = pred.detach().cpu().double(), true.cpu().double()
    return ((pred - true).norm(dim=-1) / true.norm(dim=-1)).mean().item()


MODES = [("exact", "exact"), ("exact", "learned"), ("learned", "exact"), ("learned", "learned")]
results: dict[str, dict[str, float]] = {}
preds: dict[str, dict[str, torch.Tensor]] = {}
for name, (ds, t) in tests.items():
    u_true = exact_kdv_solution(ds.slots, x, t)
    results[name], preds[name] = {}, {"truth": u_true}
    for fm, im in MODES:
        op.set_mode(fm, im)
        with torch.no_grad():
            u = op(ds.u0, t)
        results[name][f"{fm}/{im}"] = rel_l2(u, u_true)
        preds[name][f"{fm}/{im}"] = u.detach().cpu()
    with torch.no_grad():
        u = fno(ds.u0.to(dev, torch.float32), t.to(dev, torch.float32))
    results[name]["FNO"] = rel_l2(u, u_true)
    preds[name]["FNO"] = u.cpu()

cols = [f"{a}/{b}" for a, b in MODES] + ["FNO"]
print("\n== relative L2 error of u(x,t)   (encoder/decoder: exact or learned)")
print(f"{'test set':24s}" + "".join(f"{c:>17s}" for c in cols))
for name, r in results.items():
    print(f"{name:24s}" + "".join(f"{r[c]:17.2e}" for c in cols))

# encoder quality on the scattering data itself (in distribution)
ds, _ = tests["in-dist (t<=2)"]
with torch.no_grad():
    pred = op.encoder(ds.u0.to(dev, torch.float32))
m = ds.slots.presence.bool()
pres_acc = ((pred.presence.cpu() > 0.5) == m).all(-1).float().mean().item()
dk = (pred.kappa.cpu().double() - ds.slots.kappa)[m].abs()
dx = (pred.x0.cpu().double() - ds.slots.x0)[m].abs()
print(f"\nencoder: soliton count correct {100 * pres_acc:.1f}%, |d kappa| median {dk.median():.1e},"
      f" |d x0| median {dx.median():.1e}")

# physics residual of each mode (in distribution)
ds, t = tests["in-dist (t<=2)"]
phys = {}
for fm, im in MODES:
    op.set_mode(fm, im)
    with torch.no_grad():
        slots = op.encode(ds.u0[:32])
        phys[f"{fm}/{im}"] = kdv_residual_loss(lambda tt: op.decode(slots.evolve(tt.to(slots.kappa.dtype))).cpu(),
                                               t[1:-1].to(slots.kappa.dtype), h).item()
print("normalised KdV residual: " + "  ".join(f"{k} {v:.1e}" for k, v in phys.items()))

# ---------------------------------------------------------------------------
# speed
# ---------------------------------------------------------------------------
def timed(fn, reps=3):
    fn()
    if dev.type == "mps":
        torch.mps.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.time()
        fn()
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.time() - t0)
    return best


B = 256
u0b = tests["in-dist (t<=2)"][0].u0[:B]
tb = torch.linspace(0, T_TRAIN, 50)
speed = {}
for fm, im in MODES:
    op.set_mode(fm, im)
    with torch.no_grad():
        speed[f"{fm}/{im}"] = timed(lambda: op(u0b, tb), reps=1 if fm == "exact" else 3)
with torch.no_grad():
    speed["FNO"] = timed(lambda: fno(u0b.to(dev, torch.float32), tb.to(dev, torch.float32)))
print(f"\n== wall clock, {B} initial conditions x {len(tb)} times ({dev}; exact parts on CPU float64)")
for k, v in speed.items():
    print(f"  {k:16s} {v:8.3f}s   ({speed['exact/exact'] / v:7.1f}x vs exact)")

# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(len(tests), 4, figsize=(20, 4 * len(tests)), constrained_layout=True)
for row, (name, (ds, t)) in enumerate(tests.items()):
    i = 0
    tr = preds[name]["truth"][i]
    ll = preds[name]["learned/learned"][i]
    fn = preds[name]["FNO"][i]
    vmax = tr.abs().max().item()
    for col, (lab, f) in enumerate((("exact IST", tr), ("neural IST (learned/learned)", ll), ("FNO", fn))):
        spacetime_heatmap(-f, x, t, axs[row, col], f"{name}: {lab}", label="-u")
        axs[row, col].images[0].set_clim(-0.1 * vmax, vmax)
    axs[row, 3].plot(x, -tr[-1], "k", lw=2.5, label="exact")
    axs[row, 3].plot(x, -ll[-1], "C1--", lw=1.6, label="neural IST")
    axs[row, 3].plot(x, -fn[-1], "C0:", lw=1.6, label="FNO")
    axs[row, 3].set_title(f"t = {t[-1]:.2f}   rel. L2: neural IST {results[name]['learned/learned']:.1e},"
                          f" FNO {results[name]['FNO']:.1e}", fontsize=9)
    axs[row, 3].legend(fontsize=8)
fig.savefig(OUT / "neural_ist.png", dpi=110)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
for hist, lab in ((h_dec, "decoder (teacher forced)"), (h_e2e, "end-to-end")):
    ax[0].semilogy([r["step"] for r in hist], [r["u_rel"] for r in hist], label=lab)
ax[0].semilogy([r["step"] for r in h_fno], [r["u_rel"] for r in h_fno], label="FNO")
ax[0].set_xlabel("step")
ax[0].set_ylabel("train rel. L2")
ax[0].legend()
ax[1].semilogy([r["step"] for r in h_enc], [r["scatter"] for r in h_enc])
ax[1].set_title("encoder scattering loss")
ax[1].set_xlabel("step")
fig.savefig(OUT / "neural_ist_training.png", dpi=110)

json.dump({"results": results, "speed_s": speed, "physics_residual": phys, "config": S,
           "encoder": {"count_acc": pres_acc, "dkappa_median": dk.median().item(), "dx0_median": dx.median().item()},
           "train_time_s": {"neural_ist": t_ours, "fno": t_fno}},
          open(OUT / "neural_ist_results.json", "w"), indent=1)
print("wrote", OUT / "neural_ist.png", OUT / "neural_ist_training.png", OUT / "neural_ist_results.json")
torch.save({"op": op.state_dict(), "fno": fno.state_dict()}, OUT / "neural_ist_weights.pt")

if not args.quick:
    r = results
    assert r["in-dist (t<=2)"]["exact/exact"] < 1e-3
    assert r["in-dist (t<=2)"]["learned/learned"] < 0.1
    # the structural claim: exact evolution in scattering space extrapolates in time
    assert r["time extrap (2<t<=4)"]["learned/learned"] < r["time extrap (2<t<=4)"]["FNO"]
