import numpy as np
import matplotlib.pyplot as plt
import glob, os, re
from config import *

data_dir = r"loss-prob-image/2.3mK-radial-20uK-init-4img"
files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
if not files:
    raise RuntimeError("No files found")

E_rec = k_beam**2 / 2

KE_all = []
V_all  = []
p_all  = []
p2_all = []

for f in files:
    if not re.search(r'traj_\d{5}_', f):
        continue

    data = np.load(f)

    KE = data["Hp_op_t_n"]                      # <KE>
    V  = data["Hxg_op_t_n"]  # <V>
    p  = data["p_op_t_n"]                      # <p>
    p2 = data["psq_op_t_n"]                    # <p^2>

    KE_all.append(KE)
    V_all.append(V)
    p_all.append(p)
    p2_all.append(p2)

    print(f"Loaded {os.path.basename(f)}")

# convert to arrays
KE_all = np.array(KE_all)
V_all  = np.array(V_all)
p_all  = np.array(p_all)
p2_all = np.array(p2_all)


# averages
KE_mid = KE_all.mean(axis=0)
KE_mean = KE_all.mean(axis=0)[:int(len(KE_mid)*0.7)]
V_mean  = V_all.mean(axis=0)[:int(len(KE_mid)*0.7)]
p_mean  = p_all.mean(axis=0)[:int(len(KE_mid)*0.7)]
p2_mean = p2_all.mean(axis=0)[:int(len(KE_mid)*0.7)]
# KE_mean = p2_mean * 2 * m

# print("Initial temp: ", KE_mean[0]/V_depth * V_depth_kelvin * 1e3, "uK")

# variance of momentum
# per-trajectory variance
# print(p_all.shape, p2_all.shape)
# exit(0)
var_each = p2_all - p_all*p_all   # shape (N_traj, Nt)
# print(var_each.shape)
# exit(0)
# mean variance
var_mean = var_each[:, :int(len(KE_mid)*0.7)].mean(axis=0)

p_mean = p_all[:, :int(len(KE_mid)*0.7)].mean(axis=0)
p2_mean = p2_all[:, :int(len(KE_mid)*0.7)].mean(axis=0)

var_ensemble = p2_mean - p_mean**2

# error bars (optional)
var_err = var_each.std(axis=0) / np.sqrt(len(var_each))
# time axis
sample = np.load(files[0])
if "t_array" in sample:
    t = sample["t_array"]
else:
    save_psis_interval = 500
    t = np.arange(len(KE_mean))  * eps

# -----------------------
# Plot: Energies
# -----------------------
plt.figure(figsize=(8,4))
# plt.plot(t, KE_mean/E_rec, label='KE')
# plt.plot(t, (V_mean + V_depth_gs)/E_rec, label='V')
# plt.plot(t, (KE_mean + V_mean + V_depth_gs)/E_rec, label='Total')
plt.plot(t, p_mean / k_beam, label='<p>')
plt.axvline(x=37.5, color='r', linestyle='--', label='end of first half pulse')

plt.xlabel("t (Γ⁻¹)")
plt.ylabel("Momentum")
plt.title(f"avg over {len(V_all)} trajectories")
plt.legend()
plt.grid()

# -----------------------
# Plot: diffusion check
# -----------------------
plt.figure(figsize=(8,4))
# plt.plot(t, var_mean / k_beam**2, label='Var(p)')
plt.plot(t, var_ensemble / k_beam**2, label='Var(p)')
plt.axvline(x=37.5, color='r', linestyle='--', label='end of first half pulse')
plt.xlabel("t (Γ⁻¹)")
plt.ylabel("Var(p) / k_beam²")
plt.title("Momentum diffusion")
plt.legend()
plt.grid()


# -----------------------
# Plot: Energies
# -----------------------
plt.figure(figsize=(8,4))
plt.plot(t, KE_mean/E_rec, label='KE')
plt.plot(t, (V_mean + V_depth_gs)/E_rec, label='V')
plt.plot(t, (KE_mean + V_mean + V_depth_gs)/E_rec, label='Total')
# plt.plot(t, p_mean / k_beam, label='<p>')
# plt.axvline(x=37.5, color='r', linestyle='--', label='end of first half pulse')

plt.xlabel("t (Γ⁻¹)")
plt.ylabel("Energy (units of Erec)")
plt.title(f"avg over {len(V_all)} trajectories")
plt.legend()
plt.grid()


plt.tight_layout()
plt.show()

# exit(0)


# import numpy as np
# import matplotlib.pyplot as plt
# import glob, os, re
# from config import *

# data_dir = r"loss-prob-image/2.3mK-radial-20uK-init-bigger"
# files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
# if not files:
#     raise RuntimeError("No files found")

# E_rec = k_beam**2 / 2

# KE_all = []
# V_all  = []
# p_all  = []
# p2_all = []
# x_all  = []
# x2_all = []

# for f in files:
#     if not re.search(r'traj_\d{5}_', f):
#         continue

#     data = np.load(f)

#     KE_all.append(data["Hp_op_t_n"])
#     V_all.append(data["Hxg_op_t_n"])
#     p_all.append(data["p_op_t_n"])
#     p2_all.append(data["psq_op_t_n"])
#     x_all.append(data["x_op_t_n"])
#     x2_all.append(data["xsq_op_t_n"])

#     print(f"Loaded {os.path.basename(f)}")

# KE_all = np.array(KE_all)
# V_all  = np.array(V_all)
# p_all  = np.array(p_all)
# p2_all = np.array(p2_all)
# x_all  = np.array(x_all)
# x2_all = np.array(x2_all)

# KE_mid = KE_all.mean(axis=0)
# KE_mean  = KE_all.mean(axis=0)[:int(len(KE_mid)*0.65)]
# print(len(KE_mean) * 0.1)
# # exit(0)
# V_mean   = V_all.mean(axis=0)[:int(len(KE_mid)*0.65)]
# p_mean   = p_all.mean(axis=0)[:int(len(KE_mid)*0.65)]
# p2_mean  = p2_all.mean(axis=0)[:int(len(KE_mid)*0.65)]
# x_mean   = x_all.mean(axis=0)[:int(len(KE_mid)*0.65)]
# x2_mean  = x2_all.mean(axis=0)[:int(len(KE_mid)*0.65)]


# plt.plot(0.1 * np.arange(0,len(KE_mean)), KE_mean/E_rec, label='KE')
# plt.plot(0.1 * np.arange(0,len(KE_mean)),(V_mean+V_depth_gs)/E_rec, label='V')
# plt.plot(0.1 * np.arange(0,len(KE_mean)), (KE_mean + V_mean + V_depth_gs)/E_rec, label='Total')
# plt.xlabel("t (Γ⁻¹)")
# plt.ylabel("Energy (units of Erec)")
# plt.title(f"avg over {len(V_all)} trajectories")
# plt.legend()
# plt.grid()
# plt.show()

# exit(0)
# exit(0)
# import matplotlib.pyplot as plt
# var_p_ensemble = p2_mean - p_mean**2
# var_x_ensemble = x2_mean - x_mean**2   # <(Δx)^2>

# sample = np.load(files[0])
# t = sample["t_array"] if "t_array" in sample else np.arange(len(KE_mean)) * eps


# plt.plot(t, x_mean / (1/k_beam), label='<x>')
# plt.show()
# exit(0)
# var_p_ensemble = p2_mean - p_mean**2
# var_x_ensemble = x2_mean - x_mean**2   # <(Δx)^2>
# # per-trajectory quantum variance (wavepacket width)
# var_x_quantum = (x2_all - x_all**2).mean(axis=0)[:int(len(KE_mid)*0.8)]

# # classical part (spread of centers across trajectories)  
# var_x_classical = var_x_ensemble - var_x_quantum

# sample = np.load(files[0])
# t = sample["t_array"] if "t_array" in sample else np.arange(len(KE_mean)) * eps

# # ---------- power-law fit for <x^2> vs t ----------
# # fit log(<x^2>) ~ n log(t) over the 'diffusive' part
# # exclude very early times where x^2 ~ 0
# mask = (t > t[len(t)//100]) & (var_x_ensemble > 0)  # skip first 10%
# log_t  = np.log(t[mask])
# log_x2 = np.log(var_x_ensemble[mask])
# p_fit  = np.polyfit(log_t[:int(len(var_x_ensemble))], log_x2 [:int(len(var_x_ensemble))], 1)
# n_fit  = p_fit[0]
# A_fit  = np.exp(p_fit[1])
# print(f"<(Δx)²> ~ t^{n_fit:.2f}")

# # ---------- plots ----------
# fig, axs = plt.subplots(1, 2, figsize=(13, 5))

# # Left: <x^2> vs t, linear scale
# ax = axs[0]
# ax.plot(t[:int(len(var_x_ensemble))], var_x_ensemble[:int(len(var_x_ensemble))] / (beam_waist)**2, color='tab:blue', label=r'$\langle(\Delta x)^2\rangle$')
# ax.set_xlabel(r't (Γ⁻¹)')
# ax.set_ylabel(r'$\langle(\Delta x)^2\rangle\ [w_{\rm 0}^{2}]$')
# ax.set_title(f'Position spread — avg over {len(x_all)} trajectories')
# ax.grid(True, alpha=0.3)
# ax.legend()
# # trap period markers: T/4, T/2, T, 2T with T = 2π/ω
# omega_trap = nu_trap_gs  # assumed angular trap frequency ω
# T_trap = 2 * np.pi / omega_trap

# for xval, lab, ls in [
#     (T_trap / 4, r"$T/4$", ":"),
#     (T_trap / 2, r"$T/2$", "--"),
#     (T_trap,     r"$T$",   "-."),
#     (2 * T_trap, r"$2T$",  "-"),
# ]:
#     ax.axvline(x=xval, color="tab:red", linestyle=ls, alpha=0.6, label=lab)

# ax.legend()

# # Right: log-log + fit
# ax = axs[1]
# ax.loglog(t[var_x_ensemble > 0][:int(len(var_x_ensemble))], var_x_ensemble[var_x_ensemble > 0][:int(len(var_x_ensemble))] / (beam_waist)**2,
#           color='tab:blue', label=r'$\langle(\Delta x)^2\rangle$')
# t_fit_line = t[mask]
# ax.loglog(t_fit_line,
#           A_fit * t_fit_line**n_fit / (beam_waist)**2,
#           'k--', lw=1.5, label=rf'fit ~ $t^{{{n_fit:.2f}}}$')
# for xval, lab, ls in [
#     (T_trap / 4, r"$T/4$", ":"),
#     (T_trap / 2, r"$T/2$", "--"),
#     (T_trap,     r"$T$",   "-."),
#     (2 * T_trap, r"$2T$",  "-"),
# ]:
#     ax.axvline(x=xval, color="tab:red", linestyle=ls, alpha=0.6, label=lab)

# # reference lines
# t_ref = t[mask]
# for exp, ls, lab in [(1, ':', r'$\sim t$'), (2, '--', r'$\sim t^2$'), (3, '-.', r'$\sim t^3$')]:
#     norm = var_x_ensemble[mask][len(mask)//2] / t_ref[len(mask)//2]**exp / (beam_waist)**2
#     ax.loglog(t_ref, norm * t_ref**exp, color='gray', ls=ls, alpha=0.5, label=lab)
# ax.set_xlabel(r't (Γ⁻¹)')
# ax.set_ylabel(r'$\langle(\Delta x)^2\rangle\ [w_{\rm 0}^{2}]$')
# ax.set_title(rf'log-log: $\langle(\Delta x)^2\rangle \sim t^{{{n_fit:.2f}}}$')
# ax.grid(True, which='both', alpha=0.3)
# ax.legend(fontsize=9)

# plt.suptitle(r'QM: $\langle(\Delta x)^2\rangle$ vs time', fontsize=12)
# plt.tight_layout()
# plt.savefig('xsq_vs_t.png', dpi=140)
# plt.show()
# print("saved xsq_vs_t.png")


"Plotting energetic trajs below; seems like only t<200 (steps) is trustable (ie 2000 Gamma^-1) "
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob, os, re
from config import *
from helper_funcs import init_arrays, perform_ifft

data_dir = r"loss-prob-image/2.3mK-radial-20uK-init-4img"
files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
traj_files = [f for f in files if re.search(r'traj_\d{5}_', f)]
import re

def extract_temp(f):
    m = re.search(r'_(\d+(?:\.\d+)?)uK', os.path.basename(f))
    return float(m.group(1)) if m else -np.inf

# f = "loss-prob-image/2.3mK-radial-20uK-init/traj_00001_.npz"
f = "loss-prob-image/2.3mK-radial-20uK-init-4img/traj_00353_9.0uK.npz"

import numpy as np
import os

def final_energy(f):
    try:
        d = np.load(f)

        E = d["Hp_op_t_n"] + d["Hxg_op_t_n"]

        # handle different possible shapes
        if np.ndim(E) == 0:
            val = float(E)
        elif E.ndim == 1:
            val = float(E[-1])
        else:
            val = float(np.mean(E[..., -1]))  # reduce if extra dims exist

        return val

    except Exception as e:
        print(f"Skipping {f}: {e}")
        return -np.inf


# compute energies
energies = [(f, final_energy(f)) for f in traj_files]

# sort descending
energies_sorted = sorted(energies, key=lambda x: x[1], reverse=True)

# top 10
top10 = energies_sorted[:100]

# print results
print("\nTop 10 trajectories by final energy:\n")
for i, (f, e) in enumerate(top10, 1):
    print(f"{i:2d}. E_final = {e:.6e}   file = {os.path.basename(f)}")

# exit(0)
f = top10[0][0]
print("Chosen file:", f)


### 80 / 117 step
# f = traj_files[-1]
d = np.load(f)

psi = d["psip_store"]   # (2, Np, Nt)

E = d["Hp_op_t_n"] + d["Hxg_op_t_n"] 

# plt.plot(E/V_depth)
# plt.xlabel("step"); plt.ylabel("Energy (nat units)"); plt.title("Energy vs step")
# plt.grid()
# plt.show()
# exit(0)
Np, Nt = psi.shape[1], psi.shape[2]
# print(Np, Nt)
# exit(0)
x, p, dx, dp = init_arrays(N, pmax)
p_kbeam = p / k_beam
x_w0 = x / beam_waist  # in units of w0

# precompute all frames in both spaces
prob_p_g = np.abs(psi[0])**2   # (Np, Nt)
prob_p_e = np.abs(psi[1])**2

prob_x_g = np.zeros_like(prob_p_g)
prob_x_e = np.zeros_like(prob_p_e)
for i in range(Nt):
    psix = perform_ifft(psi[:, :, i])   # (2, Np)
    prob_x_g[:, i] = np.abs(psix[0])**2
    prob_x_e[:, i] = np.abs(psix[1])**2
    if i % 50 == 0:
        print(f"ifft {i}/{Nt}")

norm_g = (prob_p_g * dp).sum(axis=0)
norm_e = (prob_p_e * dp).sum(axis=0)

fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

print((prob_p_g[:,0]/dp).shape)
# top row: momentum space
lg_p, = axs[0].plot(p_kbeam, prob_p_g[:,0]/dp, color='tab:blue', lw=1.5, label='|g⟩')
le_p, = axs[0].plot(p_kbeam, prob_p_e[:,0]/dp, color='tab:red',  lw=1.5, label='|e⟩', alpha=0.7)
axs[0].set_ylabel('|ψ(p)|²/dp'); axs[0].set_title('Momentum space')
axs[0].legend(fontsize=9); axs[0].grid(True, alpha=0.3)
axs[0].set_xlabel('p / k_beam'); axs[0].set_ylabel('|ψ(x)|²/dx')

# axs[0].set_ylim(0, max(prob_p_g.max(), prob_p_e.max())/dp * 1.2)

# # top right: norm vs step (static, not animated)
# steps = np.arange(Nt)
# axs[0,1].plot(steps, norm_g, color='tab:blue', label='|g⟩')
# axs[0,1].plot(steps, norm_e, color='tab:red',  label='|e⟩', alpha=0.7)
# axs[0,1].plot(steps, norm_g+norm_e, 'k--', alpha=0.6, label='total')
# axs[0,1].set_ylabel('norm'); axs[0,1].set_title('Norm vs step')
# axs[1,0].set_xlabel('p / k_beam'); axs[1,0].set_ylabel('|ψ(x)|²/dx')

# axs[0,1].legend(fontsize=9); axs[0,1].grid(True, alpha=0.3)
# step_line = axs[0,1].axvline(0, color='gray', ls='--', lw=1)

# bottom row: position space
lg_x, = axs[1].plot(x_w0, prob_x_g[:,0]/dx, color='tab:blue', lw=1.5, label='|g⟩')
le_x, = axs[1].plot(x_w0, prob_x_e[:,0]/dx, color='tab:red',  lw=1.5, label='|e⟩', alpha=0.7)
axs[1].set_xlabel('x / w0'); axs[1].set_ylabel('|ψ(x)|²/dx')
axs[1].set_title('Position space'); axs[1].legend(fontsize=9); axs[1].grid(True, alpha=0.3)
# axs[1].set_ylim(0, max(prob_x_g.max(), prob_x_e.max())/dx * 1.2)

# # bottom right: blank / info
# axs[1,1].axis('off')
info_text = axs[1].text(0.1, 0.6, '', transform=axs[1].transAxes, fontsize=12,
                           verticalalignment='center')

title = fig.suptitle('step 0', fontsize=12)
plt.tight_layout()

n_frames = min(200, Nt)
frame_idx = np.linspace(0, Nt-1, n_frames, dtype=int)

def update(fi):
    i = frame_idx[fi]
    lg_p.set_ydata(prob_p_g[:,i]/dp)
    le_p.set_ydata(prob_p_e[:,i]/dp)
    lg_x.set_ydata(prob_x_g[:,i]/dx)
    le_x.set_ydata(prob_x_e[:,i]/dx)
    # step_line.set_xdata([i, i])
    axs[0].relim()
    axs[0].autoscale_view()

    axs[1].relim()
    axs[1].autoscale_view()

    title.set_text(f'step {i}/{Nt-1}')
    info_text.set_text(f'norm_g = {norm_g[i]:.4f}\nnorm_e = {norm_e[i]:.4f}\ntotal  = {norm_g[i]+norm_e[i]:.4f}')
    return lg_p, le_p, lg_x, le_x, title, info_text

anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)
anim.save('psi_px_evolution.gif', writer=animation.PillowWriter(fps=12))
plt.show()
print("saved psi_px_evolution.gif")