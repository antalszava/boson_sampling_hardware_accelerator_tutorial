import numpy as np
import piquasso as pq
from numpy.linalg import svd

import numpy as np
from scipy.special import factorial, genlaguerre
from scipy.constants import pi
from scipy.linalg import sqrtm


def lowdin(P, tol=1e-12):
    # Overlap / Gram matrix (Hermitian)
    M = P.conj().T @ P

    # Eigen-decomposition of Hermitian M
    w, U = np.linalg.eigh(M)

    # Handle near-linear dependence (optional)
    keep = w > tol
    U = U[:, keep]
    w = w[keep]

    Minv_sqrt = U @ np.diag(1.0 / np.sqrt(w)) @ U.conj().T
    B = P @ Minv_sqrt

    n, k = P.shape
    mx1 = B.T.conj() @ B
    mx2 = B @ B.T.conj()
    assert np.allclose(mx1, np.identity(k))
    assert np.allclose(mx2, np.identity(n))
    return B

def eigenfunction_2d_equation17(x, y, n, m):
    """
    Equation (17): Phase-convention eigenfunction ψ_{nm}(x, y) used throughout the paper.
    ψ_{nm} = √[1/((1/2(n-m))! (1/2(n+m))!)] * r^m / √π * L^m_{(n-m)/2}(r²) * e^{-r²/2} * e^{i m φ}
    
    Converts Cartesian (x,y) to polar (r, φ) internally.
    Uses the specific phase convention from eqs (13)-(17) [web:1].
    
    Parameters:
    x, y: Cartesian coordinates
    n: principal quantum number (n >= |m|, n - m even)
    m: angular momentum quantum number (m >= 0 for this form)
    
    Returns:
    ψ_{nm}(x, y)
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    abs_m = np.abs(m)
    
    # Equation (13) from "Energy level splitting for weakly interacting bosons
    # in a harmonic trap" - https://www.arxiv.org/pdf/1903.04974
    # Factorials in denominator: (1/2(n-m))! and (1/2(n+m))!
    fact1 = factorial((n - abs_m)/2)
    fact2 = factorial((n + abs_m)/2)
    
    # Normalization factor
    norm_factor = np.sqrt(fact1 / fact2)
    
    # Laguerre polynomial L^|m|_{(n-|m|)/2}(r²)
    laguerre_arg = r**2
    L = genlaguerre((n - abs_m)/2, abs_m)(laguerre_arg)
    
    phase_factor = (-1) ** (1/2*(m-abs_m))

    psi_norm = norm_factor * (r**abs_m / np.sqrt(pi)) * L * np.exp(-r**2 / 2) * np.exp(1j * m * phi)

    # Full wavefunction per eq (13)
    psi = phase_factor * psi_norm
    
    return psi

# Assume eigenfunction_2d_equation17(x, y, n, m) is already defined elsewhere.

# ----- 1. Discretize space and build orbitals -----

def build_single_particle_orbitals(n_orbitals, positions):
    """
    positions: array of shape (n_modes, 2) with (x,y) for each spatial mode χ_j
    returns: orbital_matrix[i, j] = ψ_i(χ_j)
    """
    n_modes = positions.shape[0]
    orbital_matrix = np.zeros((n_orbitals, n_modes), dtype=complex)

    # Example: label orbitals by (n, m) along some sequence
    # You can change this to match the spectrum you want.
    quantum_numbers = [(0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (2, -2)][:n_orbitals]

    for i, (n, m) in enumerate(quantum_numbers):
        for j, (x, y) in enumerate(positions):
            orbital_matrix[i, j] = eigenfunction_2d_equation17(x, y, n, m)

    return orbital_matrix

# ----- 3. Piquasso boson-sampling circuit -----

def build_piquasso_program(unitary, input_occupation):
    """
    unitary: m×m complex array
    input_occupation: list/array of length m with photon numbers per mode
    """
    m = unitary.shape[0]
    instructions = []

    instructions.append(pq.Vacuum().on_modes(*range(m)))
    # Prepare Fock input state
    for mode, n in enumerate(input_occupation):
        if n > 0:
            instructions.append(pq.Create().on_modes(mode))

    # Apply linear interferometer corresponding to unitary
    instructions.append(pq.Interferometer(unitary))

    # Measure all modes in photon number basis
    instructions.append(pq.ParticleNumberMeasurement())

    return pq.Program(instructions=instructions)

# ----- 4. Map detection pattern to particle positions -----

def pattern_to_positions(pattern, positions):
    """
    pattern: list/array of photon numbers per mode (length m)
    positions: array (m, 2) giving (x,y) per mode
    returns: array of shape (n_bosons, 2) with positions of each boson
    collision-free assumed here (0 or 1 per mode).
    """
    idx = [i for i, n in enumerate(pattern) if n > 0]
    return positions[idx, :]

# ----- 5. Define perturbation V(X) (Efimov-like example) -----

def efimov_potential_3body(positions_xyz, C=1.0):
    """
    positions_xyz: array (3, 2) for 3 bosons in 2D (x,y).
    Uses the 1D/2D version of the Efimov-inspired potential described in the paper.
    """
    x = positions_xyz
    r12 = np.linalg.norm(x[0] - x[1])
    r13 = np.linalg.norm(x[0] - x[2])
    r23 = np.linalg.norm(x[1] - x[2])

    R2 = (2.0 / 3.0) * (r12**2 + r13**2 + r23**2)
    if R2 == 0:
        return 0.0  # hard-shell cutoff in discrete model
    return -(C + 0.25) / R2

def pad_to_square(A):
    n, m = A.shape
    N = max(n, m)
    B = np.zeros((N, N), dtype=A.dtype)
    B[:n, :m] = A
    return B

# ----- 6. Boson-sampling-assisted Monte Carlo integration -----

def boson_sampling_monte_carlo(
    positions,
    n_orbitals=3,
    input_occupation=None,
    n_samples=10_000,
    potential_fn=efimov_potential_3body,
    simulator_cls=pq.SamplingSimulator,
):
    """
    Implements the algorithm of the paper:
    - g(X) = |Ψ_0(X)|^2 encoded in boson sampler
    - h(X) = V(X) evaluated classically
    - F ≈ (1/N) ∑ h(X_i) with X_i ~ g
    """
    m = positions.shape[0]
    if input_occupation is None:
        # Example: 3 bosons in first 3 modes
        input_occupation = [1, 1, 1] + [0] * (m - 3)

    # 1) Orbitals and unitary
    orbital_matrix = build_single_particle_orbitals(n_orbitals, positions)
    orbital_matrix = pad_to_square(orbital_matrix)

    #unitary = lowdin(orbital_matrix)
    unitary, _ = np.linalg.qr(orbital_matrix, mode="complete")

    # 2) Build Piquasso program and simulator
    prog = build_piquasso_program(unitary, input_occupation)
    sim = simulator_cls()

    # 3) Sample patterns and accumulate h(X)
    values = []
    for _ in range(n_samples):
        result = sim.execute(prog)
        pattern = result.samples  # list of photon counts per mode
        X = pattern_to_positions(np.array(pattern[0]), positions)
        if len(X) != 3:
            continue  # enforce 3-boson (collision-free) sector
        values.append(potential_fn(X))

    if len(values) == 0:
        raise RuntimeError("No valid 3-boson samples collected.")
    return np.mean(values), np.std(values) / np.sqrt(len(values))

# ----- 7. Example usage -----

if __name__ == "__main__":
    # Discretize a line or 2D strip into modes
    # Example: 12 modes on x-axis, y = 0
    n_modes = 12
    xs = np.linspace(-3.0, 3.0, n_modes)
    ys = np.zeros_like(xs)
    positions = np.stack([xs, ys], axis=1)

    estimate, error = boson_sampling_monte_carlo(
        positions,
        n_orbitals=5,
        n_samples=1000,
    )
    print(f"Estimated E^(1) ≈ {estimate:.4f} ± {error:.4f}")
