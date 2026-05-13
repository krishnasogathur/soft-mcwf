

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

# print("initial energy in uK: ", (d["Hp_op_t_n"][0]) * V_depth_kelvin/V_depth * 1e6)
# exit(0)
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
from helper_funcs import build_periodic_potential
V_gs = build_periodic_potential(V_depth_gs, k_lattice_natural, x)

a = np.pi / k_lattice_natural
print(a/beam_waist)
# exit(0)
plt.plot(x_w0, V_gs)
plt.xlabel('x / w0'); plt.ylabel('V(x) (nat units)'); plt.title('Periodic potential')
plt.grid()
plt.axvline(a/2/beam_waist, color='blue', ls='--', lw=1.5, label='site boundary')
plt.axvline(-a/2/beam_waist, color='green', ls='--', lw=1.5, label='site boundary')
plt.show()

# # exit(0)
# # precompute all frames in both spaces
# prob_p_g = np.abs(psi[0])**2   # (Np, Nt)
# prob_p_e = np.abs(psi[1])**2

# prob_x_g = np.zeros_like(prob_p_g)
# prob_x_e = np.zeros_like(prob_p_e)
# for i in range(Nt):
#     psix = perform_ifft(psi[:, :, i])   # (2, Np)
#     prob_x_g[:, i] = np.abs(psix[0])**2
#     prob_x_e[:, i] = np.abs(psix[1])**2
#     if i % 50 == 0:
#         print(f"ifft {i}/{Nt}")


# --- site populations ---------------------------------------------

# prob_x_total = prob_x_g + prob_x_e   # (N, Nt)
# ------------------------------------------------------------
# AVERAGED SITE POPULATIONS OVER ALL TRAJECTORIES
# ------------------------------------------------------------

from helper_funcs import perform_ifft

# lattice spacing
a = np.pi / k_lattice_natural

# site labels for each x-grid point
site_index = np.floor((x + a/2) / a).astype(int)

sites = np.arange(site_index.min(), site_index.max()+1)
Ns = len(sites)

# accumulator
site_pops_avg = None
n_used = 0

for f in traj_files:

    try:
        d = np.load(f)

        psi = d["psip_store"]   # (2, N, Nt)
        Nt = psi.shape[2]

        # per-trajectory site populations
        site_pops = np.zeros((Ns, Nt))

        for t in range(Nt):

            psix = perform_ifft(psi[:, :, t])
            # print("norm of psix: ", np.sum(np.abs(psix)**2))
            # exit(0)
            prob = (np.abs(psix[0])**2 + np.abs(psix[1])**2) / np.linalg.norm(psix)**2

            # print(np.sum(prob))
            for si, s in enumerate(sites):
                mask = (site_index == s)
                site_pops[si, t] = np.sum(prob[mask])
        # exit(0)

        # initialize accumulator
        if site_pops_avg is None:
            site_pops_avg = np.zeros_like(site_pops)

        site_pops_avg += site_pops
        n_used += 1

        print(f"processed {os.path.basename(f)}")

    except Exception as e:
        print(f"skipping {f}: {e}")

# average
site_pops_avg /= n_used

print(f"\nAveraged over {n_used} trajectories")

# ------------------------------------------------------------
# PLOT 1: averaged site histogram at one time
# ------------------------------------------------------------

# t_plot = 200

# plt.figure(figsize=(10,4))
# plt.bar(sites, site_pops_avg[:, t_plot], width=0.8)

# plt.xlabel("site index")
# plt.ylabel("avg population")
# plt.title(f"Averaged site populations at t-index = {t_plot}")
# plt.grid(alpha=0.3)
# plt.show()

# ------------------------------------------------------------
# PLOT 2: averaged time traces
# ------------------------------------------------------------

sites_to_plot = [0, -1, -2, 1, 2]

plt.figure(figsize=(10,5))

for s in sites_to_plot:

    if s in sites:

        si = np.where(sites == s)[0][0]

        plt.plot(np.arange(site_pops_avg[si].shape[0]) * 0.1 * 375, site_pops_avg[si], label=f"site {s}")

plt.xlabel("time (Gamma^-1)")
plt.ylabel("avg population")
plt.title("Average site populations vs time")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
exit(0)
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