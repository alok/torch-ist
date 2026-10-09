"""Autodiff through the IST.

(a) Inverse problem: observe a noisy KdV snapshot u(x, t=1) that came from two
    solitons, and recover (kappa_1, kappa_2, x0_1, x0_2) by gradient descent
    through the closed-form N-soliton reconstruction + time evolution.
(b) Gradients through the *direct* transform: tune the depth A of
    u0 = -A sech^2(x) so its largest bound state is kappa = 1.5
    (exact answer A = kappa (kappa + 1) = 3.75).  d kappa / d u comes from the
    implicit-function Newton step in kdv_direct.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import torch_ist as ti
from _common import OUT

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

# ---- (a) fit soliton parameters to a snapshot -------------------------------
x = torch.linspace(-20, 30, 600)
t_obs = torch.tensor([1.0])
true = dict(kappa=torch.tensor([1.3, 0.8]), x0=torch.tensor([-6.0, 2.0]))
u_obs = ti.kdv_from_solitons(x, t_obs, true["kappa"], true["x0"])[0]
u_obs = u_obs + 0.02 * torch.randn_like(u_obs)

kappa = torch.tensor([1.0, 0.6], requires_grad=True)
x0 = torch.tensor([-4.0, 0.0], requires_grad=True)
opt = torch.optim.Adam([kappa, x0], lr=0.05)
losses = []
for it in range(400):
    opt.zero_grad()
    u = ti.kdv_from_solitons(x, t_obs, kappa, x0)[0]
    loss = ((u - u_obs) ** 2).mean()
    loss.backward()
    opt.step()
    losses.append(loss.item())
print(f"(a) learned kappa = {kappa.detach().tolist()}  true {true['kappa'].tolist()}")
print(f"    learned x0    = {x0.detach().tolist()}  true {true['x0'].tolist()}")
assert torch.allclose(kappa.detach(), true["kappa"], atol=1e-2)
assert torch.allclose(x0.detach(), true["x0"], atol=5e-2)

# ---- (b) gradient through the direct scattering transform -------------------
xd = torch.linspace(-15, 15, 401)
A = torch.tensor(2.0, requires_grad=True)
target = 1.5
opt = torch.optim.Adam([A], lr=0.1)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.97)
hist = []
for it in range(150):
    opt.zero_grad()
    data = ti.kdv_direct(-A / torch.cosh(xd) ** 2, xd, reflection=False)
    k_top = data.kappa[data.mask].max()
    loss = (k_top - target) ** 2
    loss.backward()
    opt.step()
    sched.step()
    hist.append(A.item())
print(f"(b) learned A = {A.item():.5f}   exact {target * (target + 1):.5f}")
assert abs(A.item() - 3.75) < 1e-4

# gradient check of d kappa / d A against a finite difference
def top_kappa(a: float) -> float:
    return ti.kdv_direct(-a / torch.cosh(xd) ** 2, xd, reflection=False).kappa.max().item()

Ar = torch.tensor(3.0, requires_grad=True)
ti.kdv_direct(-Ar / torch.cosh(xd) ** 2, xd, reflection=False).kappa.max().backward()
fd = (top_kappa(3.0 + 1e-4) - top_kappa(3.0 - 1e-4)) / 2e-4
exact = 1 / (2 * (0.25 + 3.0) ** 0.5)   # kappa = sqrt(1/4 + A) - 1/2
print(f"    d kappa/dA at A=3: autograd {Ar.grad.item():.6f}, finite diff {fd:.6f}, exact {exact:.6f}")
assert abs(Ar.grad.item() - exact) < 1e-4

fig, axs = plt.subplots(1, 3, figsize=(17, 4), constrained_layout=True)
axs[0].plot(x, u_obs, ".", ms=2, label="noisy observation, t = 1")
axs[0].plot(x, ti.kdv_from_solitons(x, t_obs, kappa.detach(), x0.detach())[0], label="fitted 2-soliton")
axs[0].legend(); axs[0].set_title("(a) inverse problem"); axs[0].set_xlabel("x")
axs[1].semilogy(losses); axs[1].set_title("(a) loss"); axs[1].set_xlabel("Adam step")
axs[2].plot(hist); axs[2].axhline(3.75, ls="--", color="k")
axs[2].set_title(r"(b) depth $A$ so that $\kappa_{max}=1.5$"); axs[2].set_xlabel("Adam step")
fig.savefig(OUT / "learn_soliton_params.png", dpi=130)
print("wrote", OUT / "learn_soliton_params.png")
