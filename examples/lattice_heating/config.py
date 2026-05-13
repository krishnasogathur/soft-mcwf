from helper_funcs2 import init_yb_natural_units
import numpy as np

hbar = 1
Gamma = 1
m = 1
kB_actual = 1.38e-23  # in SI



params = init_yb_natural_units(Gamma)
k_beam = params["k_imaging_nat"]

N = (2**10)*8
pmax = 150*k_beam
eps = 0.1
n_steps_imaging = 5500 * 4 # 3.0 us imaging time; 1 images 
n_iters = 1000


x_scale = params["x_scale_nm"]
hbar_actual = params["hbar_actual"]
Gamma_imaging_actual = params["Gamma_imaging_actual"]
Gamma_cooling = params["Gamma_cooling_nat"]
trapping_wavelength_natural = params["trapping_wavelength_nat"]

###
lattice_wavelength_actual= 759 # in meters
lattice_wavelength_natural = lattice_wavelength_actual / x_scale # in natural units
k_lattice_natural = 2 * np.pi / lattice_wavelength_natural


# print(k_lattice_natural)

###

# Imaging parameters
s = 40
Omega_sat = (0.5 * s)**0.5 * Gamma
Omega_laser = Omega_sat
Omega_counter = Omega_sat
Delta = 0
phi = 0
se_coeff = 1 * Gamma

pulse_duration_laser = 75 * (1 / Gamma)
pulse_duration_counter = 75 * (1 / Gamma)
pulse_duration_cooling_1 = 0
pulse_duration_cooling_2 = 0
wait_time_1 = 0
wait_time_2 = 0
wait_time_3 = 0
wait_time_4 = 0

pulse_timings_1 = (
    pulse_duration_laser,
    wait_time_1,
    pulse_duration_cooling_1,
    wait_time_2,
    pulse_duration_counter,
    wait_time_3,
    pulse_duration_cooling_2,
    wait_time_4,
)


# # Trapping params
beam_waist = 580 / x_scale # in natural units
sigma_pot = beam_waist/2

vfactor = 0.220 # potential depth in mK

V_depth_kelvin = vfactor * 1e-3 # in K; chosen bc "differential light shifts can be evened across the array"
V_depth_actual = kB_actual*V_depth_kelvin
V_depth = V_depth_actual/(hbar_actual*Gamma_imaging_actual) # natural units, multiple of hbar*Gamma (since energy)


### Adding non-magic components

polarizability =  4.6 # in units of MHz / mK of ground state trap potential
dls_MHz = polarizability * vfactor # in MHz
dls = (2*np.pi*dls_MHz * 1e6) / Gamma_imaging_actual # dls in natural units (units of Gamma)


V_depth_gs = V_depth
# V_depth_es = V_depth_gs - dls # V_depth_es is less than V_depths_gs as per convention; differential light shift is the differential between ground and excited states
V_depth_es = V_depth # anti-trapped excited state

# V_depths_kelvin = np.arange(0.2, 3.2, 0.2) * 1e-3
# V_depths_actual = kB_actual*V_depths_kelvin
# V_depths = V_depths_actual/(hbar_actual*Gamma_imaging_actual) # natural units, multiple of hbar*Gamma (since energy)

# x_R = np.pi * (beam_waist**2) / trapping_wavelength_natural

# nu_trap_gs = (np.sqrt(2*V_depth_gs/m))/x_R
# nu_trap_es = (np.sqrt(2*np.abs(V_depth_es)/m))/x_R

nu_trap_gs = k_lattice_natural * np.sqrt(2*V_depth_gs/m)
nu_trap_es = nu_trap_gs
# Initial state params
mean_temp = 20 # initial atomic temperature in microkelvin; mean of a boltzmann dtbn
meanp0 = 0
alpha0 = np.sqrt(0)
x0 = np.sqrt(hbar/(m*nu_trap_gs)) #for Harmonic oscillator
p0 = hbar / x0


save_interval = 1
save_psis_interval = 375 #save every 50 Gamma^-1; going all the way up to 10,000 Gamma^-1 so it's 200 points 
# I basically downsample in time and store psips over diff iterations;
