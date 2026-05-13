import os
import sys
import time
import gc
from joblib import Parallel, delayed
from joblib import dump, load


import numpy as np
from helper_funcs import *
from config import *


x,p,dx,dp = init_arrays(N, pmax)
print(k_beam / dp)
# exit(0)
g, e = tls_basis()
sigma_gg, sigma_ee, sigma_ge, sigma_eg, I_TLS, I_space = init_operators(N, g, e)

proj_g = np.real(g @ g.T)
proj_e = np.real(e @ e.T)
exp_mikx = np.exp(-1j*k_beam*x)
exp_pikx = np.exp(1j*k_beam*x)

H_kin = build_KE_hamiltonian(m, p)
H_tls = build_tls_hamiltonian(Delta, sigma_ee) 

H_tls_block = (H_tls[None, :, :] * I_space[:, None, None])

#magic axial trap
# H_axial = build_axial_lorentzian(V_depth, beam_waist, trapping_wavelength_natural, x) # give x not x_op as its input

# V_gs = build_axial_lorentzian(V_depth_gs, beam_waist, trapping_wavelength_natural, x)
# V_es = build_axial_lorentzian(V_depth_es, beam_waist, trapping_wavelength_natural, x)

V_gs = build_gaussian_potential(V_depth_gs, sigma_pot, x)
# V_es = -build_gaussian_potential(V_depth_es, sigma_pot, x)
V_es = build_gaussian_potential(V_depth_es, sigma_pot, x) # no excited state potential. what what happes?


# V = build_harmonic_potential(m, nu_trap, x)
H_pot_block = (V_es[:, None, None] * proj_e[None, :, :] ) + (V_gs[:, None, None] * proj_g[None, :, :] )

## Laser and counter Hamiltonian terms

# for axial trap: absorption from beam is in radial direction which isn't of interest; k = 0, phases = 0 
# H_int_laser   = build_interaction_hamiltonian(Omega_laser, sigma_ge, np.ones(N))
# H_int_counter = build_interaction_hamiltonian(Omega_counter, sigma_ge, np.ones(N)) #opp dir, but here it doesn't really matter. 

H_int_laser   = build_interaction_hamiltonian(Omega_laser, sigma_ge, exp_mikx)
H_int_counter = build_interaction_hamiltonian(Omega_counter, sigma_ge, exp_pikx) #opp dir, but here it doesn't really matter.p


L_plus = np.sqrt(se_coeff / 2) * (exp_pikx[:, None, None] * sigma_ge[None, :, :])
L_minus = np.sqrt(se_coeff / 2) * (exp_mikx[:, None, None] * sigma_ge[None, :, :])
c_ops = [L_plus, L_minus]

# del L_plus, L_minus, exp_pikx, exp_mikx

correction_blocks = sum(L.conj().transpose(0, 2, 1) @ L for L in c_ops)

# --- Laser evol ops ---

H_x_laser = H_tls_block + H_int_laser + H_pot_block
H_x_nh_laser, H_p_nh_laser = init_nh_hamiltonians(H_x_laser, H_kin, correction_blocks)

del H_int_laser, H_x_laser


print("initialized Laser non-hermitian Hamiltonians")

laser_ops = prepare_nh_evol_ops(H_x_nh_laser, H_p_nh_laser, eps, hbar)
print("laser evol ops ready")

del H_x_nh_laser, H_p_nh_laser

# --- Counter evol ops ---
H_x_counter = H_tls_block + H_int_counter + H_pot_block

H_x_nh_counter, H_p_nh_counter = init_nh_hamiltonians(H_x_counter, H_kin, correction_blocks)
print("initialized Counter non-hermitian Hamiltonians")

del H_int_counter, H_x_counter

counter_ops = prepare_nh_evol_ops(H_x_nh_counter, H_p_nh_counter, eps, hbar)
print("counter evol ops ready")

del H_x_nh_counter, H_p_nh_counter



'''
PE: Harmonic, Gaussian, Lorentzian 

build_harmonic_potential: for a harmonic trap 
build_gaussian_potential: for Radial trap (Gaussian)
build_axial_lorentzian: for axial trap (Lorentzian)

general method for non-magic traps:
# H_axial_tot = np.diag(proj_e) @ H_axial_es + np.diag(proj_g) @ H_axial_gs 
'''

'''
For studying a radial trap:
# H_int_laser   = build_interaction_hamiltonian(Omega_laser, sigma_ge, exp_mikx)
'''



print(f"Input Parameters:")
print(f"  Γ (Gamma)           =     {Gamma}")
print(f"  Ω_l (Omega_laser)   =     {Omega_laser}")
print(f"  Ω_c (Omega_counter) =     {Omega_counter}")
print(f"  Δ (Delta)           =     {Delta}")
print(f"  ϕ (phase)           =     {phi}")
print(f"  ν_trap_gs           =     {nu_trap_gs} (natural)")
print(f"  ν_trap_es           =     {nu_trap_es} (natural)")
print(f"  ν_trap_gs           =     {nu_trap_gs * Gamma_imaging_actual/2/np.pi * 1e-3:.3f} (kHz)")
print(f"  ν_trap_es           =     {nu_trap_es * Gamma_imaging_actual/2/np.pi * 1e-3:.3f} (kHz)")
print(f"  V_depth_gs          =     {V_depth_gs * hbar_actual * Gamma_imaging_actual /kB_actual * 1e3:.3f} (mK)")
print(f"  V_depth_es          =     {V_depth_es * hbar_actual * Gamma_imaging_actual /kB_actual * 1e3:.3f} (mK)")
print(f"  laser pulse dur.    =     {pulse_duration_laser}      # (multiple of Γ⁻¹)")
print(f"  counter pulse dur.  =     {pulse_duration_counter}    # (multiple of Γ⁻¹)")
print(f"  ϵ (eps)             =     {eps}       # time step for evolution")
print(f"  n_steps_imaging     =     {n_steps_imaging}   # number of time steps")
print(f"  n_iters             =     {n_iters}   # number of iterations for averaging")

# exit(0)
e_ops = [x, x*x, p, p*p, V_gs, V_es, H_kin, proj_e, proj_g] # all 1D arrays of length N  
mom_array = [False, False, True, True, False, False, True, True, True]


# Save big constants once
# dump(psi_init, 'psi_init.joblib')
dump(laser_ops, 'laser_ops.joblib')
dump(counter_ops, 'counter_ops.joblib')
dump(c_ops, 'c_ops.joblib')
dump(e_ops, 'e_ops.joblib')

save_interval = 1
save_psis_interval = 375 #save every 37.5 Gamma^-1; going all the way up to 10,000 Gamma^-1 so it's 200 points 
# I basically downsample in time and store psips over diff iterations;


def run_trajectory(iter_idx):

    mean_temp = 2*2 # in checked that this dtbn sampled gives init temp of 20uK roughly

    init_temp = np.random.exponential(mean_temp)  # sampled initial temperature in microkelvin
    stdevp0 = np.sqrt(init_temp/0.3)* p0 / (vfactor**0.25) # calibrated so as to maintain initial state temperature of 20uK

    psi_tls_init = g
    psi_init, _ = build_displaced_vac_state(
        psi_tls_init, meanp0, stdevp0, alpha0, x0, p, dp, hbar
    )
    '''  
    Measuring initial temperature
    '''

    arr = compute_expectations(psi_init, [H_kin], [True])
    energies = np.sum(arr)/V_depth

    print(f"Initial temperature: {(energies)*V_depth_kelvin*1e6:.1f} uK")

    # exit(0)
    # psi_init = load('psi_init.joblib', mmap_mode='r')
    laser_ops = load('laser_ops.joblib', mmap_mode='r')
    counter_ops = load('counter_ops.joblib', mmap_mode='r')
    c_ops = load('c_ops.joblib', mmap_mode='r')
    e_ops = load('e_ops.joblib', mmap_mode='r')

    # Imaging Stage
    psi_final, (avgs_img, norms_arr_img, jump_times_img, psip_store_img) = simulate_trajectory_block(
        psi_init, eps, pulse_timings_1,
        laser_ops, counter_ops,
        None, None,
        None,
        c_ops, None, 
        e_ops, mom_array,
        n_steps_imaging,
        hbar,
        return_psis=True,
        save_interval=save_interval,
        save_psis_interval=save_psis_interval
    )


    # pop_es = es_projs_img / norms_arr_img[np.newaxis, ::save_interval]
    psip_store = psip_store_img
    x_op_t_n   = np.real(avgs_img[0, :])
    xsq_op_t_n = np.real(avgs_img[1, :])
    p_op_t_n   = np.real(avgs_img[2, :])
    psq_op_t_n = np.real(avgs_img[3, :])
    Hxg_op_t_n = np.real(avgs_img[4, :])
    Hxe_op_t_n = np.real(avgs_img[5, :])
    Hp_op_t_n  = np.real(avgs_img[6, :])
    proj_e_t_n = np.real(avgs_img[7, :])
    proj_g_t_n = np.real(avgs_img[8, :])

    norms_arr = norms_arr_img
    jumps = np.array(jump_times_img)

    # Save each result immediately
    np.savez(
        os.path.join(output_dir, f"traj_{iter_idx:05d}_{init_temp:.1f}uK.npz"),
        # pop_es = pop_es,
        psip_store = psip_store,
        # psi_final = psi_final,
        x_op_t_n = x_op_t_n,
        xsq_op_t_n = xsq_op_t_n,
        p_op_t_n = p_op_t_n,
        psq_op_t_n = psq_op_t_n,
        Hxg_op_t_n = Hxg_op_t_n,
        # Hxe_op_t_n = Hxe_op_t_n,
        Hp_op_t_n = Hp_op_t_n,
        proj_g_t_n = proj_g_t_n,
        proj_e_t_n = proj_e_t_n,
        jumps = jumps,
        norms = norms_arr,
        N=N,
        pmax=pmax,
        k_beam=k_beam,
        n_steps_imaging=n_steps_imaging,
        eps=eps,
        V_depth_gs=V_depth_gs,
        V_depth_es = V_depth_es,

    )
        # Explicitly delete references
    del psip_store, jumps, norms_arr_img, x_op_t_n, xsq_op_t_n, p_op_t_n, psq_op_t_n, Hxg_op_t_n, Hp_op_t_n, psi_final
    gc.collect()  # Force garbage collection; might not actually be doing anything but good practice

    print(f"Finished iteration {iter_idx+1}/{n_iters}", flush=True)



os.makedirs('loss-prob-image', exist_ok=True)
output_dir = f"loss-prob-image/{vfactor:.1f}mK-radial-20uK-init-4img"
os.makedirs(output_dir, exist_ok=True)



start_time = time.time()


Parallel(n_jobs=-1, backend="loky")(
    delayed(run_trajectory)(iter_idx) for iter_idx in range(n_iters)
)

end_time = time.time()

print(f"time taken to run program: {end_time - start_time}")
