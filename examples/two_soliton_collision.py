"""KdV two-soliton collision and the nonlinear phase shift.

A fast soliton (kappa_1 = 1.2) overtakes a slow one (kappa_2 = 0.7).  They
emerge with unchanged shapes but displaced:  the fast one jumps forward by
    log((k1+k2)/(k1-k2)) / k1,
the slow one backward by
    log((k1+k2)/(k1-k2)) / k2.
We start from a sampled initial profile, run the full IST pipeline (direct ->
evolve -> inverse), measure both shifts from the solution, and compare with
the formula and with a pseudo-spectral PDE solve.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import torch_ist as ti
from torch_ist.visualize import compare_solutions, spacetime_heatmap
from _common import OUT

torch.set_default_dtype(torch.float64)
k1, k2 = 1.2, 0.7
x = torch.linspace(-30, 50, 1600)
t = torch.linspace(0, 8, 321)

# initial profile: exact two-soliton at t = 0, fast one far behind the slow one
u0 = ti.kdv_from_solitons(x, torch.tensor([0.0]), [k1, k2], [-14.0, -2.0])[0]

data = ti.kdv_direct(u0, x)
print(f"recovered kappa = {data.kappa.tolist()}   (true {k1}, {k2})")
print(f"max |R(k)| = {data.R.abs().max():.2e}")
u = ti.kdv_evolve_and_invert(data.reflectionless(), x, t)
u_pde = ti.kdv_spectral_solve(u0, x, t, dt=5e-4)
err = (u - u_pde).abs().max().item()
print(f"max |IST - PDE| = {err:.2e}")


def troughs(row: np.ndarray) -> list[tuple[float, float]]:
    """(position, depth) of local minima deeper than 0.5, sub-grid by parabolic fit."""
    out = []
    xs = x.numpy()
    h = xs[1] - xs[0]
    for i in range(1, len(row) - 1):
        if row[i] < row[i - 1] and row[i] <= row[i + 1] and row[i] < -0.5:
            a, b, c = row[i - 1], row[i], row[i + 1]
            out.append((xs[i] + 0.5 * h * (a - c) / (a - 2 * b + c), -b))
    return out


un, tn = u.numpy(), t.numpy()
tracks = {"fast": ([], []), "slow": ([], [])}
for ti_, row in zip(tn, un):
    tr = troughs(row)
    if len(tr) == 2:                          # well separated: classify by depth
        for pos, depth in tr:
            key = "fast" if depth > 0.5 * (2 * k1**2 + 2 * k2**2) else "slow"
            tracks[key][0].append(ti_)
            tracks[key][1].append(pos)

t_coll = 12.0 / (4 * (k1**2 - k2**2))
shifts = {}
for key, kap in (("fast", k1), ("slow", k2)):
    tt, xx = map(np.array, tracks[key])
    v = 4 * kap**2
    before = tt < t_coll - 2.5
    after = tt > t_coll + 2.5
    # intercepts of x - v t before/after the collision
    shifts[key] = np.mean(xx[after] - v * tt[after]) - np.mean(xx[before] - v * tt[before])
L = np.log((k1 + k2) / (k1 - k2))
print(f"fast soliton shift: measured {shifts['fast']:+.4f}, theory {+L / k1:+.4f}")
print(f"slow soliton shift: measured {shifts['slow']:+.4f}, theory {-L / k2:+.4f}")
assert abs(shifts["fast"] - L / k1) < 1e-2 and abs(shifts["slow"] + L / k2) < 1e-2
assert err < 1e-3

fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
spacetime_heatmap(-u, x, t, ax, "KdV two-soliton collision via IST  ($-u$)", label="-u")
for key, kap, col in (("fast", k1, "c"), ("slow", k2, "w")):
    tt, xx = map(np.array, tracks[key])
    b = tt < t_coll - 2.5
    x_free = np.mean(xx[b] - 4 * kap**2 * tt[b]) + 4 * kap**2 * tn
    ax.plot(x_free, tn, ls="--", color=col, lw=1, label=f"{key}: no-interaction path")
ax.set_xlim(x[0].item(), x[-1].item())
ax.legend(loc="upper left")
ax.text(0.98, 0.04, f"phase shifts: fast {shifts['fast']:+.3f} (theory {L / k1:+.3f}),"
        f" slow {shifts['slow']:+.3f} (theory {-L / k2:+.3f})",
        transform=ax.transAxes, ha="right", color="w", fontsize=9)
fig.savefig(OUT / "two_soliton_collision.png", dpi=130)
compare_solutions(x, t, -u, -u_pde, "Two-soliton collision: IST vs pseudo-spectral", "-u",
                  path=OUT / "two_soliton_collision_vs_pde.png")
print("wrote", OUT / "two_soliton_collision.png")
