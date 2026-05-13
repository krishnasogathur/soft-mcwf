# import numpy as np
# import matplotlib.pyplot as plt
# import glob, os, re
# from config import *

# data_dir = r"lattice1D/corrected-0.2mK-Vg"
# files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
# if not files:
#     raise RuntimeError("No files found")

# E_rec = k_beam**2 / 2

# KE_all = []
# V_all  = []
# p_all  = []
# p2_all = []

# for f in files:
#     if not re.search(r'traj_\d{5}_', f):
#         continue

#     data = np.load(f)

#     KE = data["Hp_op_t_n"]                      # <KE>
#     V  = data["Hxg_op_t_n"]  # <V>
#     p  = data["p_op_t_n"]                      # <p>
#     p2 = data["psq_op_t_n"]                    # <p^2>

#     KE_all.append(KE)
#     V_all.append(V)
#     p_all.append(p)
#     p2_all.append(p2)

#     print(f"Loaded {os.path.basename(f)}")

# # convert to arrays
# KE_all = np.array(KE_all)
# V_all  = np.array(V_all)
# p_all  = np.array(p_all)
# p2_all = np.array(p2_all)


# # averages
# KE_mid = KE_all.mean(axis=0)
# KE_mean = KE_all.mean(axis=0)[:int(len(KE_mid)*0.7)]
# V_mean  = V_all.mean(axis=0)[:int(len(KE_mid)*0.7)]
# p_mean  = p_all.mean(axis=0)[:int(len(KE_mid)*0.7)]
# p2_mean = p2_all.mean(axis=0)[:int(len(KE_mid)*0.7)]
# # KE_mean = p2_mean * 2 * m

# # print("Initial temp: ", KE_mean[0]/V_depth * V_depth_kelvin * 1e3, "uK")

# # variance of momentum
# # per-trajectory variance
# # print(p_all.shape, p2_all.shape)
# # exit(0)
# var_each = p2_all - p_all*p_all   # shape (N_traj, Nt)
# # print(var_each.shape)
# # exit(0)
# # mean variance
# var_mean = var_each[:, :int(len(KE_mid)*0.7)].mean(axis=0)

# p_mean = p_all[:, :int(len(KE_mid)*0.7)].mean(axis=0)
# p2_mean = p2_all[:, :int(len(KE_mid)*0.7)].mean(axis=0)

# var_ensemble = p2_mean - p_mean**2

# # error bars (optional)
# var_err = var_each.std(axis=0) / np.sqrt(len(var_each))
# # time axis
# sample = np.load(files[0])
# if "t_array" in sample:
#     t = sample["t_array"]
# else:
#     save_psis_interval = 500
#     t = np.arange(len(KE_mean))  * eps

# # -----------------------
# # Plot: Energies
# # -----------------------
# plt.figure(figsize=(8,4))
# # plt.plot(t, KE_mean/E_rec, label='KE')
# # plt.plot(t, (V_mean + V_depth_gs)/E_rec, label='V')
# # plt.plot(t, (KE_mean + V_mean + V_depth_gs)/E_rec, label='Total')
# plt.plot(t, p_mean / k_beam, label='<p>')
# plt.axvline(x=37.5, color='r', linestyle='--', label='end of first half pulse')

# plt.xlabel("t (Γ⁻¹)")
# plt.ylabel("Momentum")
# plt.title(f"avg over {len(V_all)} trajectories")
# plt.legend()
# plt.grid()

# # -----------------------
# # Plot: diffusion check
# # -----------------------
# plt.figure(figsize=(8,4))
# # plt.plot(t, var_mean / k_beam**2, label='Var(p)')
# plt.plot(t, var_ensemble / k_beam**2, label='Var(p)')
# plt.axvline(x=37.5, color='r', linestyle='--', label='end of first half pulse')
# plt.xlabel("t (Γ⁻¹)")
# plt.ylabel("Var(p) / k_beam²")
# plt.title("Momentum diffusion")
# plt.legend()
# plt.grid()


# # -----------------------
# # Plot: Energies
# # -----------------------
# plt.figure(figsize=(8,4))
# plt.plot(t, KE_mean/E_rec, label='KE')
# plt.plot(t, (V_mean + V_depth_gs)/E_rec, label='V')
# plt.plot(t, (KE_mean + V_mean + V_depth_gs)/E_rec, label='Total')
# # plt.plot(t, p_mean / k_beam, label='<p>')
# # plt.axvline(x=37.5, color='r', linestyle='--', label='end of first half pulse')

# plt.xlabel("t (Γ⁻¹)")
# plt.ylabel("Energy (units of Erec)")
# plt.title(f"avg over {len(V_all)} trajectories")
# plt.legend()
# plt.grid()


# plt.tight_layout()
# plt.show()
# # exit(0)


"Plotting energetic trajs below; seems like only t<200 (steps) is trustable (ie 2000 Gamma^-1) "
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob, os, re
from config import *
from helper_funcs import init_arrays, perform_ifft

data_dir = r"lattice1D/corrected-0.2mK-Vg"
files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
traj_files = [f for f in files if re.search(r'traj_\d{5}_', f)]
import re

def extract_temp(f):
    m = re.search(r'_(\d+(?:\.\d+)?)uK', os.path.basename(f))
    return float(m.group(1)) if m else -np.inf

# f = "loss-prob-image/2.3mK-radial-20uK-init/traj_00001_.npz"
f = "lattice1D/corrected-0.2mK-Vg.npz"

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
top10 = energies_sorted[:1000]

# print results
print("\nTop 10 trajectories by final energy:\n")
for i, (f, e) in enumerate(top10, 1):
    print(f"{i:2d}. E_final = {e:.6e}   file = {os.path.basename(f)}")

# exit(0)
f = top10[999][0]
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


# Rebuild fig with 3 subplots instead of 2
plt.close('all')
from helper_funcs import build_periodic_potential
fig, axs = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
V_gs = build_periodic_potential(V_depth_gs, k_lattice_natural, x)

# ── reproduce your existing axes content ────────────────────────────────────

# top: momentum space
lg_p, = axs[0].plot(p_kbeam, prob_p_g[:,0]/dp, color='tab:blue', lw=1.5, label='|g⟩')
le_p, = axs[0].plot(p_kbeam, prob_p_e[:,0]/dp, color='tab:red',  lw=1.5, label='|e⟩', alpha=0.7)
axs[0].set_ylabel('|ψ(p)|²/dp'); axs[0].set_title('Momentum space')
axs[0].set_xlabel('p / k_beam'); axs[0].legend(fontsize=9); axs[0].grid(True, alpha=0.3)

# middle: position space
ax1_twin = axs[1].twinx()
lg_x, = axs[1].plot(x_w0, prob_x_g[:,0]/dx, color='tab:blue', lw=1.5, label='|g⟩')
v_x,  = ax1_twin.plot(x_w0, V_gs, color='tab:green', lw=1.5, label='V(x)', alpha=0.7)
le_x, = axs[1].plot(x_w0, prob_x_e[:,0]/dx, color='tab:red',  lw=1.5, label='|e⟩', alpha=0.7)
axs[1].set_xlabel('x / w0'); axs[1].set_ylabel('|ψ(x)|²/dx')
axs[1].set_title('Position space'); axs[1].legend(fontsize=9); axs[1].grid(True, alpha=0.3)
info_text = axs[1].text(0.1, 0.6, '', transform=axs[1].transAxes, fontsize=12,
                        verticalalignment='center')

# ── NEW: ball on hill ────────────────────────────────────────────────────────

x_op  = d["x_op_t_n"]         # <x> shape (Nt,)
V_op  = d["Hxg_op_t_n"]  +  0*d["Hp_op_t_n"]        # <V> shape (Nt,)
x_op_w0 = x_op / beam_waist    # rescale same as x_w0


# plt.plot(x_op_w0, V_op)
# plt.show()
# exit(0)
# static potential curve (analytic V at each x)
axs[2].plot(x_w0, V_gs, 'k-', lw=1.5, alpha=0.4, label='V(x)')
axs[2].set_xlabel('<x> / w0'); axs[2].set_ylabel('Energy (nat. units)')
axs[2].set_title('Ball on hill')
axs[2].legend(fontsize=9); axs[2].grid(True, alpha=0.3)

# trail line (last N_TRAIL frames)
N_TRAIL = 30
trail_line, = axs[2].plot([], [], 'r-', alpha=0.4, lw=1.5)

# moving dot — position = <x>, height = <V>
ball_dot, = axs[2].plot([x_op_w0[0]], [V_op[0]], 'ro', ms=12, zorder=5, label='atom')
axs[2].legend(fontsize=9)

# ── update function (replaces your existing one) ─────────────────────────────

title = fig.suptitle('step 0', fontsize=12)
plt.tight_layout()

n_frames  = min(200, Nt)
frame_idx = np.linspace(0, Nt-1, n_frames, dtype=int)

def update(fi):
    i = frame_idx[fi]

    # momentum space
    lg_p.set_ydata(prob_p_g[:,i]/dp)
    le_p.set_ydata(prob_p_e[:,i]/dp)
    axs[0].relim(); axs[0].autoscale_view()

    # position space
    lg_x.set_ydata(prob_x_g[:,i]/dx)
    le_x.set_ydata(prob_x_e[:,i]/dx)
    axs[1].relim(); axs[1].autoscale_view()
    # info_text.set_text(
    #     f'norm_g = {norm_g[i]:.4f}\nnorm_e = {norm_e[i]:.4f}\n'
    #     f'total  = {norm_g[i]+norm_e[i]:.4f}')

    # ball on hill
    ball_dot.set_data([x_op_w0[i*375]], [V_op[i*375]])
    # print(f"Values for sanity: x_op_w0[{i*375}] = {x_op_w0[i*375]}, V_op[{i*375}] = {V_op[i*375]}")
    i0 = max(0, i*375 - N_TRAIL)
    trail_line.set_data(x_op_w0[i0:i*375+1], V_op[i0:i*375+1])
    # axs[2].relim(); axs[2].autoscale_view()

    title.set_text(f'step {i}/{Nt-1}')
    return lg_p, le_p, lg_x, le_x, ball_dot, trail_line, title, info_text

anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)
anim.save('psi_px_evolution.gif', writer=animation.PillowWriter(fps=12))
plt.show()
print("saved psi_px_evolution.gif")