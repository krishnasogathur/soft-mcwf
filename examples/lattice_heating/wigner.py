"""
Compute and plot the ensemble-averaged Wigner function from psi(x,t) trajectories.
Outputs:
 - 3 static plots: t=0, t=middle, t=final (each is W(x,p) heatmap + negativity value)
 - 1 animation: W(x,p) evolving over all time
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob, os, re
from config import *
from helper_funcs import init_arrays, perform_ifft

data_dir = r"lattice1D/0.2mK-Vg"
files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
traj_files = [f for f in files if re.search(r'traj_\d{5}_', f)]
if not traj_files:
    raise RuntimeError("No traj files found")

x, p, dx, dp = init_arrays(N, pmax)

sample = np.load(traj_files[0])
psi_shape = sample["psip_store"].shape
Nt_full = psi_shape[2]
Nt = int(0.8 * Nt_full)
print(f"Nt_full={Nt_full}, using Nt={Nt}")

# downsample x grid for Wigner (N=4096 -> N_W=256)
N_W = 256
sub_idx = np.linspace(0, len(x)-1, N_W, dtype=int)
x_W = x[sub_idx]
dx_W = x_W[1] - x_W[0]
p_W = np.fft.fftshift(np.fft.fftfreq(N_W, d=dx_W) * 2*np.pi)
dp_W = p_W[1] - p_W[0]

# time snapshots — for animation use ~30 frames evenly spaced
n_anim_frames = 30
t_anim_idx = np.linspace(0, Nt-1, n_anim_frames, dtype=int)
# 3 static plots indices: 0, middle, last (within 0.8*Nt)
t_static_idx = [0, Nt//2, Nt-1]
all_t_idx = sorted(set(list(t_anim_idx) + list(t_static_idx)))

def wigner_from_psi(psi_x, x_grid, dx_grid):
    """Wigner W(X,P)=(1/pi)∫dy psi*(X+y)psi(X-y) exp(2iPy), via FFT for each X."""
    Nx = len(psi_x)
    W = np.zeros((Nx, Nx), dtype=np.float64)
    psi_re = psi_x.real; psi_im = psi_x.imag
    for i, X in enumerate(x_grid):
        Xpy = X + x_grid; Xmy = X - x_grid
        psi_p = np.interp(Xpy, x_grid, psi_re) + 1j*np.interp(Xpy, x_grid, psi_im)
        psi_m = np.interp(Xmy, x_grid, psi_re) + 1j*np.interp(Xmy, x_grid, psi_im)
        A = np.conj(psi_p) * psi_m
        Wrow = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A))) * dx_grid / np.pi
        W[i, :] = Wrow.real
    return W

N_traj_use = min(1000, len(traj_files))
print(f"Averaging Wigner over {N_traj_use} trajectories at {len(all_t_idx)} time points")

# accumulate Wigner over trajectories at each time index
W_all = {ti: np.zeros((N_W, N_W)) for ti in all_t_idx}
for tidx, fpath in enumerate(traj_files[:N_traj_use]):
    d = np.load(fpath)
    psi_p = d["psip_store"]
    for ti in all_t_idx:
        psi_p_step = psi_p[:, :, ti]
        psi_x_step = perform_ifft(psi_p_step)
        psi_g_x = psi_x_step[0][sub_idx]; psi_e_x = psi_x_step[1][sub_idx]
        norm = (np.abs(psi_g_x)**2 + np.abs(psi_e_x)**2).sum() * dx_W
        if norm > 1e-12:
            psi_g_x /= np.sqrt(norm); psi_e_x /= np.sqrt(norm)
        W_all[ti] += wigner_from_psi(psi_g_x, x_W, dx_W) + wigner_from_psi(psi_e_x, x_W, dx_W)
    if tidx % 10 == 0:
        print(f"  traj {tidx}/{N_traj_use}")
for ti in all_t_idx:
    W_all[ti] /= N_traj_use

# negativity for each
neg_all = {ti: 0.5*(np.abs(W_all[ti]).sum() - W_all[ti].sum())*dx_W*dp_W for ti in all_t_idx}

# steps to Gamma^-1
t_per_step = 0.1  * 20 # from your analyse-traj
# common color scale across all time points (use max abs of last frame for consistency)
vmax_global = max(np.max(np.abs(W_all[ti])) for ti in all_t_idx)

# common axes for plots — restrict p range to ±half (where most weight is), keep full x
p_show = 0.5   # show ±50% of p_max
ix_lo, ix_hi = 0, N_W
ip_mask = np.where(np.abs(p_W/k_beam) < p_show * (pmax/k_beam))[0]
ip_lo, ip_hi = ip_mask[0], ip_mask[-1]+1

extent = [x_W[0]/beam_waist, x_W[-1]/beam_waist,
          p_W[ip_lo]/k_beam, p_W[ip_hi-1]/k_beam]

def plot_W(ax, W_full, vmax=None, title=None):
    W = W_full[ix_lo:ix_hi, ip_lo:ip_hi]
    if vmax is None: vmax = np.max(np.abs(W))
    # symmetric log-like enhancement of small fringes via signed sqrt
    W_disp = np.sign(W) * np.sqrt(np.abs(W)/vmax) * vmax
    im = ax.imshow(W_disp.T, origin='lower', extent=extent, aspect='auto',
                   cmap='seismic', vmin=-vmax, vmax=vmax,
                   interpolation='bilinear')
    ax.set_xlabel('x / w₀'); ax.set_ylabel('p / k_beam')
    if title: ax.set_title(title)
    return im

# ---------- STATIC PLOTS ----------
labels = ['initial', 'middle', 'final']
for label, ti in zip(labels, t_static_idx):
    fig, ax = plt.subplots(figsize=(10, 5))   # wider
    W = W_all[ti]
    im = plot_W(ax, W, title=f'Wigner W(x,p) — {label}\n'
                f'step {ti} (t = {ti*t_per_step:.1f} Γ⁻¹)\n'
                f'negativity = {neg_all[ti]:.4f}')
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='W (signed √)')
    plt.tight_layout()
    fname = f'wigner_{label}.png'
    plt.savefig(fname, dpi=140)
    print(f"saved {fname}")
    plt.close()

# ---------- ANIMATION ----------
fig, ax = plt.subplots(figsize=(10, 5))   # wider
W0 = W_all[t_anim_idx[0]]
im = plot_W(ax, W0, vmax=vmax_global)
title = ax.set_title(f'step {t_anim_idx[0]}, neg = {neg_all[t_anim_idx[0]]:.3f}')
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='W (signed √)')

def update(fi):
    ti = t_anim_idx[fi]
    W = W_all[ti][ix_lo:ix_hi, ip_lo:ip_hi]
    W_disp = np.sign(W) * np.sqrt(np.abs(W)/vmax_global) * vmax_global
    im.set_data(W_disp.T)
    title.set_text(f't = {ti*t_per_step:.1f} Γ⁻¹  (step {ti})  |  neg = {neg_all[ti]:.4f}')
    return im, title

anim = animation.FuncAnimation(fig, update, frames=n_anim_frames, interval=200, blit=False)
anim.save('wigner_evolution.gif', writer=animation.PillowWriter(fps=6))
plt.close()
print("saved wigner_evolution.gif")