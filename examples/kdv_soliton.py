"""Single KdV soliton u(x,0) = -2 sech^2(x)  (u_t - 6 u u_x + u_xxx = 0).

Direct transform -> one bound state kappa = 1, c^2 = 2, R = 0.
Evolve -> c^2 e^{8t}.  Inverse -> -2 sech^2(x - 4t): a soliton moving right at speed 4.
Compared against the exact solution and a pseudo-spectral PDE solve.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import torch_ist as ti
from torch_ist.visualize import compare_solutions, spacetime_heatmap
from _common import OUT

torch.set_default_dtype(torch.float64)
x = torch.linspace(-15, 35, 1000)
t = torch.linspace(0, 5, 101)
u0 = -2 / torch.cosh(x) ** 2

data = ti.kdv_direct(u0, x)
print(f"bound states kappa = {data.kappa.tolist()}, c^2 = {data.c2.tolist()}")
print(f"max |R(k)| = {data.R.abs().max():.2e}   (reflectionless up to discretisation)")

u_ist = ti.kdv_evolve_and_invert(data.reflectionless(), x, t)
u_exact = ti.kdv_soliton(x, t, kappa=1.0)
u_pde = ti.kdv_spectral_solve(u0, x, t, dt=5e-4)

err_exact = (u_ist - u_exact).abs().max().item()
err_pde = (u_ist - u_pde).abs().max().item()
peak = x[u_ist[-1].argmin()].item()
print(f"max |IST - exact| = {err_exact:.2e}")
print(f"max |IST - PDE|   = {err_pde:.2e}")
print(f"peak at t=5: x = {peak:.3f}  (expected 20)")
assert err_exact < 1e-4 and err_pde < 1e-3 and abs(peak - 20) < 0.06

fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
spacetime_heatmap(-u_ist, x, t, ax, r"KdV soliton via IST: $-u(x,t)$, $u_0=-2\,\mathrm{sech}^2 x$", label="-u")
ax.plot(4 * t, t, "w--", lw=1, label="x = 4t")
ax.legend(loc="upper left")
fig.savefig(OUT / "kdv_soliton.png", dpi=130)
compare_solutions(x, t, -u_ist, -u_pde, "KdV single soliton: IST vs pseudo-spectral", "-u",
                  path=OUT / "kdv_soliton_vs_pde.png")
print(f"wrote {OUT/'kdv_soliton.png'} and {OUT/'kdv_soliton_vs_pde.png'}")
