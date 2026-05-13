import numpy as np
from scipy.optimize import brentq
from scipy.linalg import expm
from scipy.special import hermite, factorial


'''   
Helper functions for the block-diag implementation
'''


'''   

Initializations

'''

def init_yb_physical_vals():
    hbar = 1.055e-34          # Reduced Planck constant in J·s
    # Gamma_imaging = 2 * np.pi * 29.1e6  # Natural linewidth of 1S0 -> 1P1 (broad) transition in Hz
    Gamma_556 = 2 * np.pi * 183e3  # Natural linewidth of 1S0 -> 3P1 (narrow) transition in Hz; used for imaging and cooling
    Gamma_imaging = Gamma_556
    Gamma_cooling = 2 * np.pi * 183e3  # Natural linewidth of 1S0 -> 3P1 (narrow) transition for resolved sideband cooling; possible that it has low scattering rate due to small Gamma 
    Gamma_clock = 2 * np.pi * 7e-3     # Natural linewidth of 1S0 -> 3P0 ultranarrow Clock transition
    mass = 2.838e-25          # Atomic mass in kg (Yb-171)
    imaging_wavelength  = 556
    trapping_wavelength = 532
    cooling_wavelength  = 556
    return hbar, mass, Gamma_imaging, Gamma_cooling, Gamma_clock, (imaging_wavelength, trapping_wavelength, cooling_wavelength)


def init_yb_natural_units(Gamma_imaging_nat):

    hbar, mass, Gamma_imaging, Gamma_cooling, Gamma_clock, (imaging_wavelength, trapping_wavelength, cooling_wavelength) = init_yb_physical_vals()
    # length scale (nm)
    x_scale = np.sqrt(hbar / (mass * Gamma_imaging)) * 1e9

    # convert to natural units
    imaging_wavelength_natural  = imaging_wavelength / x_scale
    trapping_wavelength_natural = trapping_wavelength / x_scale
    cooling_wavelength_natural  = cooling_wavelength / x_scale

    # corresponding wavevectors
    k_beam  = 2 * np.pi / imaging_wavelength_natural
    k_beam_trapping = 2 * np.pi / trapping_wavelength_natural
    k_beam_cooling  = 2 * np.pi / cooling_wavelength_natural

    # scale linewidths by imaging linewidth
    Gamma_cooling_nat = Gamma_imaging_nat * (Gamma_cooling / Gamma_imaging) # since Gamma_imaging is set to 1
    Gamma_clock_nat   = Gamma_imaging_nat * (Gamma_clock / Gamma_imaging)

    return {
        "x_scale_nm": x_scale,
        "imaging_wavelength_nat": imaging_wavelength_natural,
        "trapping_wavelength_nat": trapping_wavelength_natural,
        "cooling_wavelength_nat": cooling_wavelength_natural,
        "k_imaging_nat": k_beam,
        "k_trapping_nat": k_beam_trapping,
        "k_cooling_nat": k_beam_cooling,
        "Gamma_cooling_nat": Gamma_cooling_nat,
        "Gamma_clock_nat": Gamma_clock_nat,
        "hbar_actual": hbar,
        "mass_actual": mass,
        "Gamma_imaging_actual": Gamma_imaging
    }



''' 

Demolished qutip_funcs; migrated important functions here. gotten rid of qutip altogether.

'''

def init_arrays(N, pmax): #no numpy here
    p = np.linspace(-pmax,pmax,N)
    # p = np.arange(-pmax, pmax, dp)
    x = np.fft.fftfreq(N, d=p[1]-p[0]) * 2 * np.pi 
    x = np.fft.fftshift(x) 
    dx = (x[1]-x[0])

    # p = np.fft.fftshift(np.fft.fftfreq(N, d=dx) * 2 * np.pi) 

    dp = p[1]-p[0]

    return x, p, dx, dp


def tls_basis():
    g = np.array([0,1]).reshape(-1, 1)
    e = np.array([1,0]).reshape(-1, 1)
    return g, e

def init_operators(N,g,e):
    sigma_gg = np.outer(g,g)
    sigma_ee = np.outer(e,e)
    sigma_ge = np.outer(g,e)
    sigma_eg = np.outer(e,g)
    I_TLS = np.eye(2)
    I_space = np.ones(N)
    return sigma_gg, sigma_ee, sigma_ge, sigma_eg, I_TLS, I_space


'''   

Main function base for evolution

'''


def perform_fft(psi_x_in, batched=False):
    """Transform from position to momentum basis.
    
    Input shape:
        - if batched: (2N, ...) where ... can be any shape
        - else:       (2N,)
    """

    N = psi_x_in.shape[1]

    if batched:
       
        # Apply FFT along the pos axis (axis=1)
        psi_p_out = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(psi_x_in, axes=1), axis=1),
            axes=1
        ) / np.sqrt(N)
        
        # Reshape back to original shape (2N, ...)
        return psi_p_out
    
    else:
        psi_p_out = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(psi_x_in, axes=1), axis=1),
            axes=1
        ) / np.sqrt(N)
        return psi_p_out


def perform_ifft(psi_p_in, batched=False):
    """Transform from momentum to position basis (with correct shifting).
    Input shape:
        - if batched: (2N, T)
        - else:       (2N,)
    """
    N = psi_p_in.shape[1]

    if batched:
        psi_x_out = np.fft.fftshift(
            np.fft.ifft(np.fft.ifftshift(psi_p_in, axes=1), axis=1),
            axes=1
        ) * np.sqrt(N)
        # return psi_x_out.reshape(2 * N, -1)
        return psi_x_out
    else:
        psi_x_out = np.fft.fftshift(
            np.fft.ifft(np.fft.ifftshift(psi_p_in, axes=1), axis=1),
            axes=1
        ) * np.sqrt(N)
        return psi_x_out



'''   

even faster implementation: do everything in pspace bc momentum operator doesn't require mat/vec muls at all

'''

# def compute_expectations(psi, ops, mom_space, N_psi): #along with ops array, I also provide a bool array of whether op is in mom space.

#     if len(ops) == 0:
#         return None
#     psi_expec = psi/np.linalg.norm(psi) # psi needs to be norm 1
    
#     if not(np.all(mom_space)):
#         psi_expec_pos = perform_ifft(psi_expec, N_psi)
#         psi_prob_pos = np.sum(np.abs(psi_expec_pos)**2, axis=0)

#     psi_prob = np.sum(np.abs(psi_expec)**2, axis=0)

#     vals = np.zeros(len(ops))  # or float if all ops are Hermitian

#     for i, op in enumerate(ops):  
#         if mom_space[i]==True:
#             val = np.sum(op * psi_prob)
#             vals[i] = np.real(val)
#         else:
#             val = np.sum(op * psi_prob_pos)
#             vals[i] = np.real(val)

#     return vals

def compute_expectations(psi, ops, mom_space):
    psi = psi / np.linalg.norm(psi)

    if not np.all(mom_space):
        psi_pos = perform_ifft(psi)   # (2,N)

    vals = np.zeros(len(ops))

    for i, op in enumerate(ops):
        if op.ndim == 1:
            # Oscillator operator (diag length-N)
            wf = psi if mom_space[i] else psi_pos   # (2,N)
            op_psi = wf * op                       # multiply each row by diag
            vals[i] = np.vdot(wf, op_psi).real     # sum over both TLS blocks

        elif op.shape == (2,2):
            # TLS operator (2x2)
            op_psi = op @ psi                      # acts on TLS index
            vals[i] = np.vdot(psi, op_psi).real    # full Hilbert inner product

        else:
            raise ValueError(f"Unsupported op shape {op.shape}")

    return vals





'''  
EVOLUTION OPERATORS:
'''

'''  
Two options for evol_ops:

1. using expm to compute matrix exponentials of each 2x2 block (accurate but slower for large N)
2. eigendecompose each 2x2 matrix and use that to compute exp(-iHt); involves 2 additional matmuls from V and Vdag but these are 2x2 matmuls 

'''


def evol_ops_expm(H_blocks, dt, hbar=1.0):
    """
    H_blocks: (N,2,2) array (block-diagonal Hamiltonian)
    dt: evolution timestep
    hbar: Planck's constant (default=1.0)
    
    returns U_blocks of shape (N,2,2)
    """
    Nidx = H_blocks.shape[0]
    U_blocks = np.empty_like(H_blocks, dtype=np.complex128)
    for n in range(Nidx):
        U_blocks[n] = expm(-1j * H_blocks[n] * dt / hbar)
        
    return U_blocks



def evol_ops(t, H_x_nh_eigs_block, H_p_nh_eigs, hbar=1.0):
    """
    Build evolution operators from precomputed eigenvalues/eigenvectors.

    H_x_nh_block : (N,2) array of eigenvalues of each 2x2 block of Non-hermitian hamilotnian
    """
    
    U_x_nh_eigs_block = np.exp(-1j * H_x_nh_eigs_block * t / hbar)   # shape (2,N)
    U_p_nh_eigs = np.exp(-1j * H_p_nh_eigs * t / (2*hbar))


    return U_x_nh_eigs_block, U_p_nh_eigs


'''  
Deciding which approach to use and subsequently building evol_ops appropriately:

By default, we'd prefer the approach using eigendecomposition since it's faster for large N (that's what evol_ops does, subsequently to be called in perform_soft_trajectories)

So the natural evol_ops (eigendecomp version) is the one used for brentq; eigenvecs V and Vdag are precomputed and passed to sub_step_evol to reconstruct the effect of
evolution operator on the state (via perform_soft_trajectories) (surprisingly it's only 4x faster than expm approach and is indep of N; understandable bc one involves
)


However, for general U_eps (which is fixed in time and can hence be precomputed), we can use either the expm approach or the eigendecomp approach 
but we'd like to directly compute the block diagonal U_eps with dims (N,2,2) since it is a one-time reconstruction. The advantage of using this 
in perform_soft_trajectories is that it is much faster due to parallelization of np.einsum() 

Also tested using random numbers that both approaches match and that using U_eps is faster in the long run that using eigendecomp approach for a fixed time step.


TL;DR:
1. for U_eps, we directly precompute U_eps (N,2,2) and use that in perform_soft_trajectories
2. for a general U(t) (to be passed to brentq, we use eigendecomp approach since it is faster for large N and we can precompute V and Vdag)
''' 

### Helper funcs for block eigendecomposition and preparing evol ops
def block_eigendecomp(H_blocks):
    """
    Perform eigendecomposition of a block-diagonal Hamiltonian.

    Parameters
    ----------
    H_blocks : ndarray, shape (N,2,2)
        Block-diagonal Hamiltonian (2x2 per block)

    Returns
    -------
    eigvals : ndarray, shape (N,2)
        Eigenvalues of each 2x2 block
    V : ndarray, shape (N,2,2)
        Eigenvectors of each 2x2 block
    V_inv : ndarray, shape (N,2,2)
        Inverse of eigenvectors for each block
    """
    N = H_blocks.shape[0]
    eigvals = np.empty((2,N), dtype=np.complex128)
    V = np.empty((N,2,2), dtype=np.complex128)
    V_inv = np.empty((N,2,2), dtype=np.complex128)

    for n in range(N):
        vals, vecs = np.linalg.eig(H_blocks[n])
        eigvals[:, n] = vals
        V[n] = vecs
        V_inv[n] = np.linalg.inv(vecs)

    # thought about the order of eigenvals for a while; it makes most sense to store it as 2,N (as is our initial state psi)

    return eigvals, V, V_inv 


def prepare_nh_evol_ops(H_x_nh_block, H_p_nh_diag, eps, hbar):
    """
    Given non-Hermitian position and momentum Hamiltonians, compute the
    evolution operators and return relevant decompositions.

    Returns the list:
            [U_x_nh_eps, U_p_nh_eps,
             H_x_nh_eigs, V_x_nh, V_inv_x_nh,
             H_p_nh_diag]
    """
    # Eigendecomposition
    H_x_nh_eigs_block, V_x_nh, V_inv_x_nh = block_eigendecomp(H_x_nh_block)

    # H_p_nh_eigs: assumed to be 1D array (diag of the matrix)

    # Time evolution operators using the user-defined evol_ops function
    U_x_nh_eps_eigs_block, U_p_nh_eps_eigs = evol_ops(eps, H_x_nh_eigs_block, H_p_nh_diag, hbar) 

    # Reconstruct full U_x_nh_eps from eigendecomposition
    U_x_nh_eps_block = np.einsum('nij,jn,njk->nik', V_x_nh, U_x_nh_eps_eigs_block, V_inv_x_nh) 
    # above einsum is non-trivial actually; lemme bc I want to maintain eigs as 2,N for easy element-wise product

    '''  
    U_x_nh_eps (Nx2x2), U_p_nh_eps_eigs (Nx1): used in the general SOFT step (perform_soft_trajectories)
    H_x_nh_eigs, V_x_nh, V_inv_x_nh, H_p_nh_diag: used in sub_step_evol for brentq input
    '''
    return [U_x_nh_eps_block, U_p_nh_eps_eigs,
            H_x_nh_eigs_block, V_x_nh, V_inv_x_nh,
            H_p_nh_diag]


def act_block_on_state(U_blocks, psi):
    """
    Apply a block-diagonal operator to a state vector.

    Parameters
    ----------
    U_blocks : ndarray, shape (N,2,2)
        Block-diagonal operator (2x2 per block)
    psi : ndarray, shape (2,N)
        State vector

    Returns
    -------
    psi_out : ndarray, shape (2,N)
        Resulting state vector after applying U_blocks
    """
    psi_out = np.einsum('nij,jn->in', U_blocks, psi)
    return psi_out
    

def perform_soft_trajectories(psi_in, U_x_nh_eps, U_x_nh_eigs, V_x, V_inv_x, U_p_diag, optim=True):
   
    psiout = psi_in
    # Half potential:
    psiout = U_p_diag[None, :] * psiout # already matches along the motional dimension

    psi_ifft = perform_ifft(psiout)

    if optim==False:
        psi_ifft = act_block_on_state(U_x_nh_eps, psi_ifft)  # if not optimizing, I simply act the pre-constructed U_x_nh_eps
    else:
        psi_ifft = act_block_on_state(V_inv_x, psi_ifft)       # Transform to eigenbasis
        psi_ifft = U_x_nh_eigs * psi_ifft        # Both U_x_nh_eigs and psi_ifft are 2xN so elementwise multiplication works
        psi_ifft = act_block_on_state(V_x, psi_ifft) 

    psiout = perform_fft(psi_ifft)

    # Half potential the second time
    psiout = U_p_diag[None, :] * psiout        

    return psiout


'''  
I've hardcoded perform_soft_trajs to only perform eigendecomp throughout optimization process;
only uses eps when optim=False which is the usual case (everything else involves sub_step_evol
which requires evaluating at arbitrary times)
'''

def sub_step_evol(t, psi, r, U_x_nh_eps, H_x_nh_eigs, V_x_nh, V_inv_x_nh, H_p_nh_eigs, hbar=1, return_norm=False): 
    U_x_nh_eigs, U_p_nh_eigs = evol_ops(t, H_x_nh_eigs, H_p_nh_eigs, hbar ) # if necessary, switch to linear later 
    psi_out = perform_soft_trajectories(psi, U_x_nh_eps, U_x_nh_eigs, V_x_nh, V_inv_x_nh, U_p_nh_eigs, optim=True) # all evol ops are diagonal, so multiplication is easie

    if return_norm==True:
        return np.linalg.norm(psi_out)**2-r
    return psi_out


def find_jump_time(psi_prev, eps, r,  U_x_nh_eps, H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar=1):
    """
    Finds the sub-time t* ∈ [0, eps] where the evolved psi_prev's norm matches r.
    """
    # print("prev norm inside find jump time func:", np.linalg.norm(psi_prev) **2)

    val0 = sub_step_evol(0, psi_prev, r, U_x_nh_eps,  H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar , True ) 
    val1 = sub_step_evol(eps, psi_prev, r, U_x_nh_eps,  H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar , True )

    if val0 * val1 > 0:
        raise ValueError("Jump not bracketed in [0, eps] — this shouldn't happen if norm detection was correct.")

    t_star = brentq(sub_step_evol, 0, eps, args=(psi_prev, r, U_x_nh_eps,  H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar , True ), xtol=1e-10)
    return t_star

 
def perform_jump(psi, c_ops):

   
    # (since jump operator is in position space we first convert)
    psiout = perform_ifft(psi)

    # Instead of computing matmul twice, I will do it once and just store the psis. seems much faster this way
    # I'll just candidates + their squared norms in one pass
    psi_candidates = []
    norms = []

    for op in c_ops:
        candidate_state = act_block_on_state(op, psiout)
        norm_candidate = np.linalg.norm(candidate_state)
        psi_candidates.append(candidate_state)
        norms.append(norm_candidate)
    
    norms = np.array(norms)
    probs = norms**2
    probs /= probs.sum()

  
    # Choose jump according to probs
    chosen_index = np.random.choice(len(c_ops), p=probs)

    # Normalize chosen candidate with precomputed norm
    psi = psi_candidates[chosen_index] / norms[chosen_index]
    
    # Transform back to momentum space 
    psiout = perform_fft(psi)
    return psiout


def detect_jump_and_update(psi_curr, eps, r, U_x_nh_eps, H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, c_ops, hbar=1):
    t_star = find_jump_time(psi_curr, eps, r,  U_x_nh_eps, H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar)
    # print(f"Jump at time step {i}, t* = {t_star:.4e}")
    psi_updated = sub_step_evol(t_star, psi_curr ,r, U_x_nh_eps, H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar) #this is the time at which it gets projected, hence we evolve until norm reaches that point
    psi_after_jump = perform_jump(psi_updated, c_ops)

    return psi_after_jump, t_star
 

def finish_eps_evol(del_t, psi_after_jump, r, U_x_nh_eps, H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar=1):
    psi_eps = sub_step_evol(del_t, psi_after_jump, r, U_x_nh_eps, H_x_nh_eigs, H_x_nh_V, H_x_nh_Vinv, H_p_nh_eigs, hbar) # evolving the updated state for rest of the time, given by eps-t_curr
    return psi_eps


### Build init state

def build_fock_state(psi_tls, n_fock, nu_trap, xi, dx, m=1, hbar=1): # by default init in momentum space
    
    def fock_n(n, x, nu_trap, m=1, hbar=1):
        
        """Returns the n-th Fock state position space wavefunction for QHO."""
        coeff = ((m * nu_trap / (np.pi * hbar)) ** (1 / 4)) / np.sqrt((2 ** n) * factorial(n))
        poly = hermite(n)
        gaussian = np.exp(-x ** 2 / 2)
        return coeff * poly(x) * gaussian
    
    
    psi_x_init = fock_n(n_fock, xi, nu_trap, m , hbar)  # position space wavefunction in the n-th Fock state
    psi_x_init = psi_x_init * np.sqrt(dx) #this is the psi which satisfies sum(abs(psi)^2) = 1, and useful to preserve normalization in our wavefunction

    psi_tls = psi_tls.ravel()
    psi_tls = psi_tls / np.linalg.norm(psi_tls)  # normalized; so that sum(psi(x)^2 dx) = 1 (integral as a discrete sum)
    psi_init_x = np.kron(psi_x_init, psi_tls)  # |g⟩ ⊗ |ψ(x)⟩ #positional state tensored with atomic state for complete state description 


    psi_init_x = psi_init_x.reshape(2,-1, order='F') # need a 2,N array whose 0 corresponds to e and 1 corresponds to g; np.kron() in opp dir messes this up
    psi_init_x /= np.linalg.norm(psi_init_x)  # Normalizing the combined state, so that sum(abs(psi)^2) = 1 (integral as a discrete sum)

    psi_init = perform_fft(psi_init_x)

    return psi_init, psi_init_x


def build_displaced_vac_state(psi_tls, meanp0, stdevp0, alpha0, x0, p, dp, hbar=1.0):
    """
    Constructs the full initial wavefunction: |ψ⟩ =  |ψ(p)⟩ ⊗  |tls⟩,
    where |ψ(p)⟩ is a momentum-space Gaussian optionally displaced by alpha0.

    Returns:
        psi_init : np.ndarray
            Normalized full wavefunction in momentum space (flattened)
        psi_init_xspace : np.ndarray
            Position-space wavefunction (via IFFT)
    """
    # Normalize TLS state
    psi_tls = psi_tls.ravel()
    psi_tls = psi_tls / np.linalg.norm(psi_tls)

    # Gaussian ground state in momentum space
    pi = (p - meanp0) / stdevp0
    ground_state = (1 / (np.pi * stdevp0**2))**0.25 * np.exp(-pi**2 / 2)

     
    psi_p_init = (ground_state.astype(complex)) * np.exp(-1j * (np.sqrt(2) * np.real(alpha0) * x0 / hbar) * p) # alpha0: gives displacement in x space
    # Normalize discrete state
    psi_p_init = psi_p_init * np.sqrt(dp)

    # # Tensor product: |tls⟩ ⊗ |ψ⟩
    # psi_init = np.kron(psi_tls, psi_p_init).reshape(-1)
    # psi_init = psi_init / np.linalg.norm(psi_init)

    psi_init = psi_p_init[None, :] * psi_tls[:, None]  # shape (2,N)
    psi_init = psi_init / np.linalg.norm(psi_init)

    psi_init_x = perform_ifft(psi_init)
    
    return psi_init, psi_init_x



## Helper funcs for constructing diff Hamiltonian terms and evol. ops

def build_KE_hamiltonian(mass, p):
    KE = (p**2 / (2 * mass))
    return KE


def build_tls_hamiltonian(Delta, sigma_ee):
    """
    Constructs the two-level system (TLS) internal Hamiltonian:
        H_tls = Δ * |e⟩⟨e| ⊗ I

    By definition, Δ = w_atom - w_laser so >0 implies red detuned beam. 
    """
    return Delta * sigma_ee


def build_gaussian_potential(V_depth, sigma_trap, x):
    """
    Constructs a Gaussian potential Hamiltonian:
        V(x) = -V₀ * exp(-x² / (2σ²)), where σ = (1/ν) * sqrt(V₀ / m)
    """
    V = -V_depth * np.exp(-x**2 / (2 * sigma_trap**2))
    return V


def build_axial_lorentzian(V_depth, beam_waist, trapping_wavelength, x):
    """
    Constructs the axial Lorentzian potential from a Gaussian beam intensity profile:
        V(x) = -V₀ / (1 + (x / x_R)^2), 
    where x_R = π * w0^2 / λ is the Rayleigh range.
    """
    # Rayleigh range
    x_R = np.pi * beam_waist**2 / trapping_wavelength

    # Lorentzian axial profile
    V = -V_depth / (1 + (x / x_R)**2)

    # Build Hamiltonian on total Hilbert space
    return V


def convert_nu_sigma_gaussian(*, V_depth, m=1, value, direction="nu->sigma"):
    """
    Converts between trap frequency around trap center (nu_trap) and potential width (sigma_pot),
    using nu_trap = (2 * sqrt(V_depth / m)) / beam_waist and sigma = beam_waist / 2.
    """
    if direction == 'nu->sigma':
        nu = value
        beam_waist = (2 * np.sqrt(V_depth / m)) / nu
        sigma = beam_waist / 2
        return sigma
    
    elif direction == 'sigma->nu':
        sigma = value
        beam_waist = 2 * sigma
        nu = (2 * np.sqrt(V_depth / m)) / beam_waist
        return nu
    
    else:
        raise ValueError("Invalid direction. Use 'nu->sigma' or 'sigma->nu'.")


def build_harmonic_potential(mass, nu_trap, x):
    """
    Constructs a harmonic potential Hamiltonian:
        V(x) = (1/2) * m * ω² * x²
    """
    V = 0.5 * mass * ((nu_trap)**2) * (x**2)
    return V


def build_interaction_hamiltonian(Omega, sigma_ge, exp_ikx, phase=0.0):
    """
    Constructs the Hermitian laser interaction Hamiltonian:
        H_int = - (Omega / 2) * kron(sigma_ge, exp_ikx) + h.c.

    In block-diagonal scheme, it represents H_int as an (N, 2, 2) array, where each slice H_int[n] is the
        2x2 block corresponding to position index n.
        """
    prefactor = -(Omega / 2) * np.exp(-1j * phase)

    # broadcast into (N,2,2)
    H_blocks = prefactor * exp_ikx[:, None, None] * sigma_ge[None, :, :]

    H_blocks = H_blocks + np.conjugate(np.transpose(H_blocks, (0, 2, 1)))

    return H_blocks

def init_nh_hamiltonians(H_x, H_p, correction_blocks):
    """
    Construct non-Hermitian Hamiltonians by adding -i/2 * L†L terms to H_x/.
    """
    # Build non-Hermitian contribution from collapse ops
    # correction = sum(L.conj().T @ L for L in c_ops)
    H_x_nh = H_x - (1j / 2) * correction_blocks
    
    return H_x_nh, H_p


# def simulate_trajectory_block(
#     psi_input, eps, pulse_timings,
#     laser_ops, counter_ops,
#     cooling_ops, cooling_ops_counter,
#     noint_ops,
#     c_ops_lasers, c_ops_cooling,
#     e_ops, mom_array,
#     n_steps,
#     N, hbar=1,
#     return_psis=False,
#     save_interval=1,
#     save_psis_interval=100   
# ):
#     psip = psi_input
#     # Downsampling factor for psip_store 
#     downsample_factor = save_interval
#     n_store_steps = (n_steps + downsample_factor - 1) // downsample_factor

#     downsample_factor_psi = save_psis_interval
#     n_store_steps_psi = (n_steps + downsample_factor_psi - 1) // downsample_factor_psi

#     avg_vals = np.zeros((len(e_ops) , n_store_steps))
    
#     norms_arr = np.zeros(n_store_steps)

#     jump_times = []


#     # es_projs = np.zeros((N, n_store_steps))

#     if return_psis:
#         psi_return = np.zeros(psip.shape + (n_store_steps_psi,), dtype=complex)
#     else:
#         psi_return = None

#     store_idx = 0
#     store_idx_psi = 0

#     if len(pulse_timings) != 8:
#         raise ValueError(
#             "pulse_timings must be a tuple of 8 values: "
#             "(pulse_duration_laser, wait_time_1, pulse_duration_cooling_1, wait_time_2, "
#             " pulse_duration_counter, wait_time_3, pulse_duration_cooling_2, wait_time_4)"
#         )

#     (laser_pulse, wait_time_1, cooling_pulse_1, wait_time_2,
#      counter_pulse, wait_time_3, cooling_pulse_2, wait_time_4) = pulse_timings

#     # Cumulative segment edges
#     t1 = laser_pulse
#     t2 = t1 + wait_time_1
#     t3 = t2 + cooling_pulse_1
#     t4 = t3 + wait_time_2
#     t5 = t4 + counter_pulse
#     t6 = t5 + wait_time_3
#     t7 = t6 + cooling_pulse_2
#     t8 = t7 + wait_time_4
#     cycle_time = t8

#     r = np.random.rand()  # jump threshold

#     for step in range(n_steps):

#         if step == 0 or ((step+1) % (n_steps//10) == 0):
#             print(f"Step {step+1}/{n_steps}...")

#         # Downsampled storage
#         if (step % downsample_factor == 0):
#             # es_projs[:, store_idx] = (proj_e * np.abs(psip) ** 2)[:N]
#             # Compute all expectations (every step)
#             norms_arr[store_idx] = np.linalg.norm(psip) ** 2
#             avg_vals[:, store_idx] = compute_expectations(psip, e_ops, mom_array)

#             store_idx += 1

#         if (step % downsample_factor_psi == 0):

#             if return_psis:
#                 psi_return[:, :, store_idx_psi] = psip
            
#             store_idx_psi += 1

#         # Determine which Hamiltonian and collapse operators to use
#         t_present = step * eps
#         time_in_cycle = t_present % cycle_time

#         if time_in_cycle < t1:
#             ham_ops = laser_ops
#             c_ops = c_ops_lasers
#         elif time_in_cycle < t2:
#             ham_ops = noint_ops
#         elif time_in_cycle < t3:
#             ham_ops = cooling_ops
#             c_ops = c_ops_cooling
#         elif time_in_cycle < t4:
#             ham_ops = noint_ops
#         elif time_in_cycle < t5:
#             ham_ops = counter_ops
#             c_ops = c_ops_lasers
#         elif time_in_cycle < t6:
#             ham_ops = noint_ops
#         elif time_in_cycle < t7:
#             ham_ops = cooling_ops_counter
#             c_ops = c_ops_cooling
#         else:
#             ham_ops = noint_ops

#         (U_x_nh_eps_block, U_p_nh_eps,
#          H_x_nh_eigs, V_x_nh, V_inv_x_nh,
#          H_p_nh_eigs) = ham_ops

#         # Soft propagation
#         psi_soft = perform_soft_trajectories(psip, U_x_nh_eps_block, None, None, None, U_p_nh_eps, optim=False)
#         norm_sq_next = np.linalg.norm(psi_soft) ** 2

#         t_curr = 0

#         while norm_sq_next < r:

#             # print("SE detected: ", "normsq = ", norm_sq_next, "r = ", r)

#             psi_after_jump, tstar = detect_jump_and_update(
#                 psip, eps - t_curr, r,
#                 None, # U_x_nh_eps_block not actually used during optimization
#                 H_x_nh_eigs, V_x_nh, V_inv_x_nh,
#                 H_p_nh_eigs, c_ops, hbar
#             )

#             r = np.random.rand()  # reset jump threshold
#             t_curr += tstar # update present sub-step time

#             # print("Jump detected at time = ", {step * eps + t_curr})

#             jump_times.append(step * eps + t_curr)

#             psi_eps_pred = finish_eps_evol(
#                 eps - t_curr, psi_after_jump, r,
#                 None, # also not used here
#                 H_x_nh_eigs, V_x_nh, V_inv_x_nh,
#                 H_p_nh_eigs, hbar
#             )
#             norm_eps_pred = np.linalg.norm(psi_eps_pred) ** 2

#             if norm_eps_pred >= r:
#                 psi_soft = psi_eps_pred
#                 norm_sq_next = norm_eps_pred
#             else:
#                 psip = psi_after_jump

#         psip = psi_soft

#     # Build return tuple
#     returns = [avg_vals, norms_arr, jump_times, psi_return]

#     return psip, tuple(returns)


# def simulate_trajectory_block(
#     psi_input, eps, pulse_timings,
#     laser_ops, counter_ops,
#     cooling_ops, cooling_ops_counter,
#     noint_ops,
#     c_ops_lasers, c_ops_cooling,
#     e_ops, mom_array,
#     n_steps,
#     hbar=1,
#     return_psis=False,
#     save_interval=1,
#     save_psis_interval=100   
# ):
#     psip = psi_input
#     # Downsampling factor for psip_store 
#     downsample_factor = save_interval
#     n_store_steps = (n_steps + downsample_factor - 1) // downsample_factor

#     downsample_factor_psi = save_psis_interval
#     n_store_steps_psi = (n_steps + downsample_factor_psi - 1) // downsample_factor_psi

#     avg_vals = np.zeros((len(e_ops) , n_store_steps))
    
#     norms_arr = np.zeros(n_store_steps)

#     jump_times = []


#     # es_projs = np.zeros((N, n_store_steps))

#     if return_psis:
#         psi_return = np.zeros(psip.shape + (n_store_steps_psi,), dtype=complex)
#     else:
#         psi_return = None

#     store_idx = 0
#     store_idx_psi = 0

#     if len(pulse_timings) != 8:
#         raise ValueError(
#             "pulse_timings must be a tuple of 8 values: "
#             "(pulse_duration_laser, wait_time_1, pulse_duration_cooling_1, wait_time_2, "
#             " pulse_duration_counter, wait_time_3, pulse_duration_cooling_2, wait_time_4)"
#         )

#     (laser_pulse, wait_time_1, cooling_pulse_1, wait_time_2,
#      counter_pulse, wait_time_3, cooling_pulse_2, wait_time_4) = pulse_timings

#     # Cumulative segment edges
#     t1 = laser_pulse
#     t2 = t1 + wait_time_1
#     t3 = t2 + cooling_pulse_1
#     t4 = t3 + wait_time_2
#     t5 = t4 + counter_pulse
#     t6 = t5 + wait_time_3
#     t7 = t6 + cooling_pulse_2
#     t8 = t7 + wait_time_4
#     cycle_time = t8

#     r = np.random.rand()  # jump threshold

#     for step in range(n_steps):

#         t1_step, t5_step = t1, t5
        
#         if step == 0:
#             t1_step = t1/2
#         elif step == n_steps - 1:
#             t5_step =  t5 - counter_pulse/2

#         # above ensures symmetry in first and last pulses.

#         if step == 0 or ((step+1) % (n_steps//10) == 0):
#             print(f"Step {step+1}/{n_steps}...")

#         # Downsampled storage
#         if (step % downsample_factor == 0):
#             # es_projs[:, store_idx] = (proj_e * np.abs(psip) ** 2)[:N]
#             # Compute all expectations (every step)
#             norms_arr[store_idx] = np.linalg.norm(psip) ** 2
#             avg_vals[:, store_idx] = compute_expectations(psip, e_ops, mom_array)

#             store_idx += 1

#         if (step % downsample_factor_psi == 0):

#             if return_psis:
#                 psi_return[:, :, store_idx_psi] = psip
            
#             store_idx_psi += 1

#         # Determine which Hamiltonian and collapse operators to use        

#         t_present = step * eps
#         time_in_cycle = t_present % cycle_time

#         if time_in_cycle < t1_step:
#             ham_ops = laser_ops
#             c_ops = c_ops_lasers
#         elif time_in_cycle < t2:
#             ham_ops = noint_ops
#         elif time_in_cycle < t3:
#             ham_ops = cooling_ops
#             c_ops = c_ops_cooling
#         elif time_in_cycle < t4:
#             ham_ops = noint_ops
#         elif time_in_cycle < t5_step:
#             ham_ops = counter_ops
#             c_ops = c_ops_lasers
#         elif time_in_cycle < t6:
#             ham_ops = noint_ops
#         elif time_in_cycle < t7:
#             ham_ops = cooling_ops_counter
#             c_ops = c_ops_cooling
#         else:
#             ham_ops = noint_ops

#         (U_x_nh_eps_block, U_p_nh_eps,
#          H_x_nh_eigs, V_x_nh, V_inv_x_nh,
#          H_p_nh_eigs) = ham_ops

#         # Soft propagation
#         psi_soft = perform_soft_trajectories(psip, U_x_nh_eps_block, None, None, None, U_p_nh_eps, optim=False)
#         norm_sq_next = np.linalg.norm(psi_soft) ** 2

#         t_curr = 0

#         while norm_sq_next < r:

#             # print("SE detected: ", "normsq = ", norm_sq_next, "r = ", r)

#             psi_after_jump, tstar = detect_jump_and_update(
#                 psip, eps - t_curr, r,
#                 None, # U_x_nh_eps_block not actually used during optimization
#                 H_x_nh_eigs, V_x_nh, V_inv_x_nh,
#                 H_p_nh_eigs, c_ops, hbar
#             )

#             r = np.random.rand()  # reset jump threshold
#             t_curr += tstar # update present sub-step time

#             # print("Jump detected at time = ", {step * eps + t_curr})

#             jump_times.append(step * eps + t_curr)

#             psi_eps_pred = finish_eps_evol(
#                 eps - t_curr, psi_after_jump, r,
#                 None, # also not used here
#                 H_x_nh_eigs, V_x_nh, V_inv_x_nh,
#                 H_p_nh_eigs, hbar
#             )
#             norm_eps_pred = np.linalg.norm(psi_eps_pred) ** 2

#             if norm_eps_pred >= r:
#                 psi_soft = psi_eps_pred
#                 norm_sq_next = norm_eps_pred
#             else:
#                 psip = psi_after_jump

#         psip = psi_soft

#     # Build return tuple
#     returns = [avg_vals, norms_arr, jump_times, psi_return]

#     return psip, tuple(returns)



def simulate_trajectory_block(
    psi_input, eps, pulse_timings,
    laser_ops, counter_ops,
    cooling_ops, cooling_ops_counter,
    noint_ops,
    c_ops_lasers, c_ops_cooling,
    e_ops, mom_array,
    n_steps,
    hbar=1,
    return_psis=False,
    save_interval=1,
    save_psis_interval=100   
):
    psip = psi_input
    # Downsampling factor for psip_store 
    downsample_factor = save_interval
    n_store_steps = (n_steps + downsample_factor - 1) // downsample_factor

    downsample_factor_psi = save_psis_interval
    n_store_steps_psi = (n_steps + downsample_factor_psi - 1) // downsample_factor_psi

    avg_vals = np.zeros((len(e_ops) , n_store_steps))
    
    norms_arr = np.zeros(n_store_steps)

    jump_times = []


    # es_projs = np.zeros((N, n_store_steps))

    if return_psis:
        psi_return = np.zeros(psip.shape + (n_store_steps_psi,), dtype=complex)
    else:
        psi_return = None

    store_idx = 0
    store_idx_psi = 0

    if len(pulse_timings) != 8:
        raise ValueError(
            "pulse_timings must be a tuple of 8 values: "
            "(pulse_duration_laser, wait_time_1, pulse_duration_cooling_1, wait_time_2, "
            " pulse_duration_counter, wait_time_3, pulse_duration_cooling_2, wait_time_4)"
        )

    r = np.random.rand()  # jump threshold


    (laser_pulse, wait_time_1, cooling_pulse_1, wait_time_2,
     counter_pulse, wait_time_3, cooling_pulse_2, wait_time_4) = pulse_timings

    def recompute_edges(laser_pulse_val, counter_pulse_val):
        t1 = laser_pulse_val
        t2 = t1 + wait_time_1
        t3 = t2 + cooling_pulse_1
        t4 = t3 + wait_time_2
        t5 = t4 + counter_pulse_val
        t6 = t5 + wait_time_3
        t7 = t6 + cooling_pulse_2
        t8 = t7 + wait_time_4
        return t1, t2, t3, t4, t5, t6, t7, t8, t8  # last one is cycle_time
    
    full_edges = recompute_edges(laser_pulse, counter_pulse)         
    first_edges = recompute_edges(laser_pulse/2, counter_pulse)      
    last_edges = recompute_edges(laser_pulse, counter_pulse/2)        

    full_cycle_time = full_edges[-1]    # 1.6
    first_cycle_time = first_edges[-1]  # 1.4  
    last_cycle_time = last_edges[-1]    # 1.4

    for step in range(n_steps):

        if step == 0 or ((step+1) % (n_steps//10) == 0):
            print(f"Step {step+1}/{n_steps}...")

        # Downsampled storage
        if (step % downsample_factor == 0):
            # es_projs[:, store_idx] = (proj_e * np.abs(psip) ** 2)[:N]
            # Compute all expectations (every step)
            norms_arr[store_idx] = np.linalg.norm(psip) ** 2
            avg_vals[:, store_idx] = compute_expectations(psip, e_ops, mom_array)

            store_idx += 1

        if (step % downsample_factor_psi == 0):

            if return_psis:
                psi_return[:, :, store_idx_psi] = psip
            
            store_idx_psi += 1

        ### Logic to ensure first and last pulses are shortened
        t_present = step * eps

        # Decide which edges to use
        if t_present + 1e-10 < first_cycle_time:
            t1, t2, t3, t4, t5, t6, t7, t8, cycle_time = first_edges
            t_present_cycle = t_present + 1e-10

        elif (t_present + 1e-10 > first_cycle_time + (full_cycle_time * int(np.floor((n_steps*eps - first_cycle_time) / full_cycle_time  + 1e-10)))):
            t_present_cycle = t_present - first_cycle_time  - (full_cycle_time * int(np.floor((n_steps*eps - first_cycle_time) / full_cycle_time  + 1e-10))) + 1e-10
            t1, t2, t3, t4, t5, t6, t7, t8, cycle_time = last_edges
        
        else:
            t1, t2, t3, t4, t5, t6, t7, t8, cycle_time = full_edges
            t_present_cycle = t_present - first_cycle_time + 1e-10     
            
        time_in_cycle = t_present_cycle % cycle_time
        eps_tol = 1e-10

        if time_in_cycle < t1 - eps_tol:
            ham_ops = laser_ops
            c_ops = c_ops_lasers
        elif time_in_cycle < t2 - eps_tol:
            ham_ops = noint_ops
        elif time_in_cycle < t3 - eps_tol:
            ham_ops = cooling_ops
            c_ops = c_ops_cooling
        elif time_in_cycle < t4 - eps_tol:
            ham_ops = noint_ops
        elif time_in_cycle < t5 - eps_tol:
            ham_ops = counter_ops
            c_ops = c_ops_lasers
        elif time_in_cycle < t6 - eps_tol:
            ham_ops = noint_ops
        elif time_in_cycle < t7 - eps_tol:
            ham_ops = cooling_ops_counter
            c_ops = c_ops_cooling
        else:
            ham_ops = noint_ops

        (U_x_nh_eps_block, U_p_nh_eps,
         H_x_nh_eigs, V_x_nh, V_inv_x_nh,
         H_p_nh_eigs) = ham_ops

        # Soft propagation
        psi_soft = perform_soft_trajectories(psip, U_x_nh_eps_block, None, None, None, U_p_nh_eps, optim=False)
        norm_sq_next = np.linalg.norm(psi_soft) ** 2

        t_curr = 0

        while norm_sq_next < r:

            # print("SE detected: ", "normsq = ", norm_sq_next, "r = ", r)

            psi_after_jump, tstar = detect_jump_and_update(
                psip, eps - t_curr, r,
                None, # U_x_nh_eps_block not actually used during optimization
                H_x_nh_eigs, V_x_nh, V_inv_x_nh,
                H_p_nh_eigs, c_ops, hbar
            )

            r = np.random.rand()  # reset jump threshold
            t_curr += tstar # update present sub-step time

            # print("Jump detected at time = ", {step * eps + t_curr})

            jump_times.append(step * eps + t_curr)

            psi_eps_pred = finish_eps_evol(
                eps - t_curr, psi_after_jump, r,
                None, # also not used here
                H_x_nh_eigs, V_x_nh, V_inv_x_nh,
                H_p_nh_eigs, hbar
            )
            norm_eps_pred = np.linalg.norm(psi_eps_pred) ** 2

            if norm_eps_pred >= r:
                psi_soft = psi_eps_pred
                norm_sq_next = norm_eps_pred
            else:
                psip = psi_after_jump

        psip = psi_soft

    # Build return tuple
    returns = [avg_vals, norms_arr, jump_times, psi_return]

    return psip, tuple(returns)


def long_evolve(psi, U_x_qho, U_p_qho, evol_time, eps, N_psi, save_interval=10):
    
    n_steps = int(evol_time / eps)
    n_saved = (n_steps + save_interval - 1) // save_interval  # ceil(n_steps / 10)

    psi_snapshots = np.zeros((2 * N_psi, n_saved), dtype=complex)

    psiin = psi
    snap_idx = 0

    for t_idx in range(n_steps):
        psiout = perform_soft_trajectories(psiin, U_x_qho, _, _, U_p_qho, N_psi, diag_ham=True)

        # Save every save_interval'th step
        if t_idx % save_interval == 0:
            psi_snapshots[:, snap_idx] = psiout
            snap_idx += 1

        psiin = psiout

    return psi_snapshots


def compute_scattering_rate(s, Gamma, Delta):

    return 0.5*Gamma*(s / (1 + s + (2*Delta/Gamma)**2))

'''
Above expression for scattering rate comes from solving Optical bloch eqns at steady state to get the e.s. population; this times
the decay rate gives scattering rate
'''