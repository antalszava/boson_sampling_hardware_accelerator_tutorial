import numpy as np

import piquasso as pq
from numpy.linalg import svd

import numpy as np
from scipy.special import factorial, genlaguerre
from scipy.constants import pi
from scipy.linalg import sqrtm

def lowdin(P):
    n, k = P.shape

    U, _, Vh = np.linalg.svd(P, full_matrices=False)

    b = U @ Vh

    mx1 = b.T.conj() @ b
    mx2 = b @ b.T.conj()
    if n < k:
        assert np.allclose(mx2, np.identity(n))
    elif k < n:
        assert np.allclose(mx1, np.identity(n))
    else:
        assert np.allclose(mx1, np.identity(n))
        assert np.allclose(mx2, np.identity(n))

    return b

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

    # Validation: n >= m >= 0, n - m even (as per eq 17 form)
    if m < 0 or n < m or (n - m) % 2 != 0:
        return 0.0

    # Factorials in denominator: (1/2(n-m))! and (1/2(n+m))!
    fact1 = factorial((n - m)/2)
    fact2 = factorial((n + m)/2)

    # Normalization factor from eq (17)
    norm_factor = np.sqrt(fact1 / fact2)

    # Laguerre polynomial L^m_{(n-m)/2}(r²) - no absolute value on m
    laguerre_arg = r**2
    L = genlaguerre(m, (n - m)/2)(laguerre_arg)

    # Full wavefunction per eq (17)
    psi = norm_factor * (r**m / np.sqrt(pi)) * L * np.exp(-r**2 / 2) * np.exp(1j * m * phi)

    return psi

# Assume eigenfunction_2d_equation17(x, y, n, m) is already defined elsewhere.

# ----- 1. Discretize space and build orbitals -----
def build_single_particle_orbitals(positions, quantum_numbers=None):
    """
    positions: array of shape (n_modes, 2) with (x,y) for each spatial mode χ_j
    returns: orbital_matrix[i, j] = ψ_i(χ_j)
    """
    n_orbitals = 3
    n_modes = positions.shape[0]
    orbital_matrix = np.zeros((n_orbitals, n_modes), dtype=complex)

    # Example: label orbitals by (n, m) along some sequence
    # You can change this to match the spectrum you want.
    if quantum_numbers is None:
        quantum_numbers = [(0, 0), (1, 1), (2, 2)]

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

def extend_to_orthonormal_rows(mat_with_ortogonal_rows, seed=None, tol=1e-12, max_tries=10_000):
    """

    Args:

       mat_with_ortogonal_rows: (3, 12) numpy array (real or complex). Rows assumed mutually orthogonal
           under the Hermitian inner product <u,v> = vdot(u,v).

    Returns:

      Q: (12, 12) with orthonormal rows under Hermitian inner product,
         i.e., Q @ Q.conj().T == I.
    """
    A = np.asarray(mat_with_ortogonal_rows)
    if A.shape != (3, 12):
        raise ValueError(f"Expected shape (3,12), got {A.shape}")

    # Preserve complex dtype if present; otherwise promote to complex128 safely
    if not np.iscomplexobj(A):
        A = A.astype(np.complex128)

    rng = np.random.default_rng(seed)

    basis = [A[i].copy() for i in range(3)]

    def hermitian_proj_coeff(b, w):
        # coefficient c such that w - c*b removes component along b:
        # c = <b,w>/<b,b> with Hermitian inner product implemented by vdot (conjugates first arg) [web:28]
        bb = np.vdot(b, b)
        if np.abs(bb) <= tol:
            return 0.0 + 0.0j
        return np.vdot(b, w) / bb

    def orthogonalize(v, basis_list):
        w = v.copy()
        for b in basis_list:
            w = w - hermitian_proj_coeff(b, w) * b
        return w

    tries = 0
    while len(basis) < 12:
        if tries >= max_tries:
            raise RuntimeError("Could not generate enough independent vectors; increase max_tries or check input.")
        tries += 1

        # random complex vector (real + i*imag)
        v = rng.standard_normal(12) + 1j * rng.standard_normal(12)
        w = orthogonalize(v, basis)

        if np.linalg.norm(w) > tol:
            basis.append(w)

    Q = np.stack(basis, axis=0)  # (12, 12)

    # Normalize rows at the end using the Hermitian norm ||q|| = sqrt(<q,q>) [web:28]
    row_norms = np.sqrt(np.real(np.vdot(Q, Q).reshape(1, 1)))  # not used; kept simple below

    # Better: per-row norms
    norms = np.sqrt(np.real(np.einsum('ij,ij->i', Q.conj(), Q)))
    Q = Q / norms[:, None]

    return Q

# ----- 6. Boson-sampling-assisted Monte Carlo integration -----

def boson_sampling_monte_carlo(
    positions,
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
        # 1 photon in the first 3 modes
        input_occupation = [1, 1, 1] + [0] * (m - 3)

    # 1) Orbitals and unitary
    orbital_matrix = build_single_particle_orbitals(positions)

    assert np.linalg.matrix_rank(orbital_matrix) == orbital_matrix.shape[0]

    ortogonal_rows_mat = lowdin(orbital_matrix)
    assert np.linalg.matrix_rank(ortogonal_rows_mat) == orbital_matrix.shape[0]

    unitary = extend_to_orthonormal_rows(ortogonal_rows_mat, seed=0)

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
        n_samples=10000,
    )
    print(f"Estimated E^(1) ≈ {estimate:.4f} ± {error:.4f}")
