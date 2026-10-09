"""General (non-reflectionless) KdV data: u(x,0) = -2.5 exp(-x^2).

The direct transform finds one soliton plus a nonzero reflection coefficient.
The full Marchenko inversion (continuous spectrum + low-rank soliton part)
reconstructs u(x,t) including the dispersive radiation tail, compared with a
pseudo-spectral solve.  Also shown: the soliton-only part (R dropped), which is
what the solution looks like as t -> infinity.
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import torch_ist as ti
from torch_ist.visualize import compare_solutions, plot_kdv_spectrum, spacetime_heatmap
from _common import OUT

torch.set_default_dtype(torch.float64)
x = torch.linspace(-20, 20, 321)
t = torch.linspace(0, 1.5, 31)
u0 = -2.5 * torch.exp(-x ** 2)

# kdv_ist_solve picks a spectral grid fine enough for the radiation at t_max
t0 = time.time()
u_ist, data = ti.kdv_ist_solve(u0, x, t)
print(f"direct + Marchenko inversion of {len(t)} time slices: {time.time() - t0:.1f}s  ({data.k.numel()} k-points)")
print(f"kappa = {data.kappa.tolist()}, max|R| = {data.R.abs().max():.3f}, |R(0)| = {data.R[data.k.abs().argmin()].abs():.3f}")

u_sol = ti.kdv_evolve_and_invert(data.reflectionless(), x, t)
# the PDE reference is periodic: pad the domain 6x so radiation (speed 12 k^2)
# cannot wrap around into the comparison window
h = (x[1] - x[0]).item()
pad = 2 * 320
xp = x[0] - pad * h + h * torch.arange(len(x) + 2 * pad)
u_pde = ti.kdv_spectral_solve(-2.5 * torch.exp(-xp ** 2), xp, t, dt=2e-4)[..., pad:pad + len(x)]

rt = (u_ist[0] - u0).abs().max().item()
err = (u_ist - u_pde).abs().max().item()
print(f"t=0 round trip error = {rt:.2e}")
print(f"max |IST - PDE| = {err:.2e}  (|u| up to {u0.abs().max():.1f})")
assert rt < 5e-3 and err < 5e-3

fig, axs = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
spacetime_heatmap(u_ist, x, t, axs[0], "full IST (solitons + radiation)", "RdBu_r", "u", symmetric=True)
spacetime_heatmap(u_sol, x, t, axs[1], "soliton part only (R = 0)", "RdBu_r", "u", symmetric=True)
plot_kdv_spectrum(data, axs[2])
axs[2].set_xlim(-6, 6)
axs[2].set_title("scattering data")
fig.savefig(OUT / "kdv_radiation.png", dpi=130)
compare_solutions(x, t, u_ist, u_pde, "KdV, u0 = -2.5 exp(-x^2): Marchenko IST vs pseudo-spectral", "u",
                  cmap="RdBu_r", path=OUT / "kdv_radiation_vs_pde.png", symmetric=True)
print("wrote", OUT / "kdv_radiation.png")
