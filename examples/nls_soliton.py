"""Focusing NLS  i q_t + q_xx + 2|q|^2 q = 0  via IST.

1. q(x,0) = sech(x): one eigenvalue zeta = i/2, r = 0 -> stationary soliton sech(x) e^{it}.
2. q(x,0) = 2 sech(x): zeta = i/2, 3i/2 (Satsuma-Yajima) -> periodic N=2 breather.
3. q(x,0) = 1.2 sech(x): zeta = 0.7i plus radiation (r != 0) -> full Marchenko inversion.
Each is compared against a split-step Fourier PDE solve.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import torch_ist as ti
from torch_ist.visualize import compare_solutions, spacetime_heatmap
from _common import OUT

torch.set_default_dtype(torch.float64)

# ---- 1. single soliton ------------------------------------------------------
x = torch.linspace(-20, 20, 512)
t = torch.linspace(0, 6, 121)
q0 = (1 / torch.cosh(x)).to(torch.complex128)
data = ti.nls_direct(q0, x)
print(f"[sech]   zeta = {data.zeta.tolist()}, C = {data.C.tolist()}, max|r| = {data.r.abs().max():.1e}")
q_ist = ti.nls_evolve_and_invert(data.reflectionless(), x, t)
q_exact = ti.nls_soliton(x, t)                 # sech(x) e^{it}
q_pde = ti.nls_split_step_solve(q0, x, t, dt=1e-3)
e1, e1p = (q_ist - q_exact).abs().max().item(), (q_ist - q_pde).abs().max().item()
print(f"[sech]   max|IST - exact| = {e1:.2e}, max|IST - PDE| = {e1p:.2e}")
assert e1 < 1e-4 and e1p < 1e-4

# ---- 2. N = 2 breather ------------------------------------------------------
t2 = torch.linspace(0, 6, 241)
q0b = (2 / torch.cosh(x)).to(torch.complex128)
data_b = ti.nls_direct(q0b, x)
print(f"[2 sech] zeta = {[round(z.imag, 6) for z in data_b.zeta.tolist()]} (i/2, 3i/2 expected)")
qb_ist = ti.nls_evolve_and_invert(data_b.reflectionless(), x, t2)
qb_pde = ti.nls_split_step_solve(q0b, x, t2, dt=2e-4)
e2 = (qb_ist - qb_pde).abs().max().item()
print(f"[2 sech] max|IST - PDE| = {e2:.2e}")
assert e2 < 1e-3

# ---- 3. soliton + radiation (general Marchenko) -----------------------------
xg = torch.linspace(-16, 16, 257)
t3 = torch.linspace(0, 3, 31)
q0g = (1.2 / torch.cosh(xg)).to(torch.complex128)
qg_ist, data_g = ti.nls_ist_solve(q0g, xg, t3)
print(f"[1.2sech] zeta = {data_g.zeta.tolist()}, max|r| = {data_g.r.abs().max():.3f}")
qg_pde = ti.nls_split_step_solve(q0g, xg, t3, dt=5e-4)
e3 = (qg_ist - qg_pde).abs().max().item()
print(f"[1.2sech] max|IST(Marchenko) - PDE| = {e3:.2e}")
assert e3 < 2e-2

fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
spacetime_heatmap(q_ist, x, t, axs[0], r"$q_0=\mathrm{sech}\,x$: soliton $|q|$", label="|q|")
spacetime_heatmap(qb_ist, x, t2, axs[1], r"$q_0=2\,\mathrm{sech}\,x$: N=2 breather $|q|$", label="|q|")
spacetime_heatmap(qg_ist, xg, t3, axs[2], r"$q_0=1.2\,\mathrm{sech}\,x$: soliton + radiation $|q|$", label="|q|")
for ax in axs:
    ax.set_xlim(-10, 10)
fig.suptitle("Focusing NLS via inverse scattering")
fig.savefig(OUT / "nls_soliton.png", dpi=130)
compare_solutions(x, t2, qb_ist, qb_pde, "NLS breather (2 sech x): IST vs split-step", "|q|",
                  path=OUT / "nls_breather_vs_pde.png")
compare_solutions(xg, t3, qg_ist, qg_pde, "NLS 1.2 sech x (Marchenko with radiation) vs split-step", "|q|",
                  path=OUT / "nls_radiation_vs_pde.png")
print("wrote", OUT / "nls_soliton.png")
