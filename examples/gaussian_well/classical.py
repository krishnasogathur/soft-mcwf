"""
1D semiclassical sim with 556 nm imaging beam (intercombination line).
Gamma/2pi = 183 kHz. Same pulse protocol: alternating kicks, 75 Gamma^-1 half-pulses.
Plots: survival vs time, KE/PE/Total vs time.
"""
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time

hbar = 1.054571817e-34; kB = 1.380649e-23; amu = 1.66053906660e-27
m      = 170.9363258*amu
lam    = 556e-9                       # 556 nm intercombination line
k_ph   = 2*np.pi/lam
Gamma  = 2*np.pi*183e3                # Gamma/2pi = 183 kHz
v_rec  = hbar*k_ph/m
E_rec  = 0.5*m*v_rec**2

V0_K   = 2.3e-3; V0 = V0_K*kB
sigma  = 290e-9
T0     = 20e-6; sigma_v = np.sqrt(kB*T0/m)
dt_max = 1e-9 * 29.1 / 0.183                        # longer dt ok since Gamma is small

# pulse: 75 Gamma^-1 half-pulse
pulse_Gamma = 75.0
pulse_s     = pulse_Gamma / Gamma

# total: 10 images x 150 Gamma^-1 per image
N_images    = 3
# total_Gamma = N_images * 150.0
total_Gamma = 3300 * 0.7

total_s     = total_Gamma / Gamma

print(f"556nm params:")
print(f"  Gamma/2pi = {Gamma/(2*np.pi)/1e3:.1f} kHz")
print(f"  v_rec = {v_rec*1e3:.2f} mm/s")
print(f"  E_rec/kB = {E_rec/kB*1e6:.3f} uK")
print(f"  pulse_s = {pulse_s*1e6:.1f} us")
print(f"  total_s = {total_s*1e6:.1f} us  ({total_Gamma:.0f} Gamma^-1)")

@njit(fastmath=True)
def kick_sign(t, pulse_s):
    if t < pulse_s/2:
        return 1.0
    n = int((t - pulse_s/2) // pulse_s)
    return -1.0 if (n % 2 == 0) else 1.0

@njit(fastmath=True)
def run_one(t_samples, dt_max, V0, sigma, m, Gamma, v_rec, pulse_s, sigma_v,
            KE_at, V_at, alive_at):
    n  = t_samples.shape[0]
    x  = 0.0; v = np.random.normal()*sigma_v
    t  = 0.0; s2 = sigma*sigma
    t_sc = -np.log(np.random.random())*(1/Gamma) * (1/0.48)
    idx  = 0
    while idx < n:
        t_samp   = t_samples[idx]
        is_sc    = t_sc < t_samp
        t_next   = t_sc if is_sc else t_samp
        while t < t_next:
            dt = dt_max
            if t+dt > t_next: dt = t_next - t
            expf = np.exp(-x*x/(2*s2))
            F    = -V0*(x/s2)*expf
            v   += 0.5*(F/m)*dt; x += v*dt
            expf = np.exp(-x*x/(2*s2))
            F    = -V0*(x/s2)*expf
            v   += 0.5*(F/m)*dt; t += dt
        if is_sc:
            v   += kick_sign(t, pulse_s)*v_rec
            if np.random.random() < 0.5: v += v_rec
            else:                         v -= v_rec
            t_sc = t - np.log(np.random.random())*2.0/Gamma
        else:
            ke = 0.5*m*v*v
            pe = -V0*np.exp(-x*x/(2*s2))
            KE_at[idx]    = ke
            V_at[idx]     = pe
            alive_at[idx] = 1 if (ke + pe) <= 0.0 else 0
            idx += 1

@njit(fastmath=True)
def batch(N, t_samples, dt_max, V0, sigma, m, Gamma, v_rec, pulse_s, sigma_v):
    n = t_samples.shape[0]
    KE_sum = np.zeros(n); V_sum = np.zeros(n); n_alive = np.zeros(n, dtype=np.int64)
    KE_at  = np.zeros(n); V_at  = np.zeros(n); alive_at = np.zeros(n, dtype=np.int64)
    for _ in range(N):
        for i in range(n): KE_at[i]=0.; V_at[i]=0.; alive_at[i]=0
        run_one(t_samples, dt_max, V0, sigma, m, Gamma, v_rec, pulse_s, sigma_v,
                KE_at, V_at, alive_at)
        for i in range(n):
            KE_sum[i]  += KE_at[i]; V_sum[i] += V_at[i]; n_alive[i] += alive_at[i]
    return KE_sum, V_sum, n_alive

# sample every 5 Gamma^-1
t_Gamma  = np.arange(5, total_Gamma+1, 5, dtype=float)
t_s      = t_Gamma / Gamma

# warmup
_ = batch(5, t_s[:2].copy(), dt_max, V0, sigma, m, Gamma, v_rec, pulse_s, sigma_v)

N_tr = 50000
t0   = time.time()
KE_sum, V_sum, n_alive = batch(N_tr, t_s, dt_max, V0, sigma, m, Gamma, v_rec, pulse_s, sigma_v)
print(f"done {time.time()-t0:.1f}s")

KE_mean   = KE_sum / N_tr
V_mean    = V_sum  / N_tr
PE_mean   = V_mean + V0          # shifted so bottom = 0
E_mean    = KE_mean + PE_mean
surv      = n_alive / N_tr
err_surv  = np.sqrt(np.maximum(surv*(1-surv), 1e-12)/N_tr)

# image boundaries (every 150 Gamma^-1)
img_lines = np.arange(150, total_Gamma+1, 150*3)

# real time in us
t_us = t_s * 1e6

# --- survival plot ---
fig, ax = plt.subplots(figsize=(6.5, 5))

ax.errorbar(t_Gamma, surv, yerr=err_surv, fmt='-', lw=1.5,
            color='tab:blue', ecolor='lightblue', elinewidth=0.8, label='E < V₀')

for il in img_lines:
    ax.axvline(il, color='gray', ls='--', alpha=0.4, lw=0.8)

ax.set_xlabel(r't ($\Gamma^{-1}$)')
ax.set_ylabel('Survival probability')

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(img_lines)
ax2.set_xticklabels([f'{x*1e6:.0f}' for x in img_lines/Gamma])
ax2.set_xlabel('t (μs)')

# ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.set_title('Survival (E < V₀)')
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()


# --- energy plot ---
fig, ax = plt.subplots(figsize=(6.5, 5))

ax.plot(t_Gamma, KE_mean/E_rec,  label=r'$\langle KE\rangle$', color='tab:blue')
ax.plot(t_Gamma, PE_mean/E_rec,  label=r'$\langle V\rangle$',  color='tab:orange')
ax.plot(t_Gamma, E_mean/E_rec,   label=r'$\langle E_{\rm tot}\rangle$', color='k', lw=2)

# ax.axhline(V0/E_rec, color='red', ls='--', alpha=0.6, label='V₀')

for il in img_lines:
    ax.axvline(il, color='gray', ls='--', alpha=0.4, lw=0.8)

ax.set_xlabel(r't ($\Gamma^{-1}$)')
ax.set_ylabel(r'Energy / $E_{\rm rec}$')
ax.set_title('Energy evolution')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()