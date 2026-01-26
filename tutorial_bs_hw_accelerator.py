"""Boson-sampling Monte Carlo accelerator tutorial.

This script accompanies the manuscript "Experimental demonstration of boson
sampling as a hardware accelerator for Monte Carlo integration" and provides a
Piquasso-based implementation of the core algorithm.
"""

import itertools

import numpy as np
import piquasso as pq
from scipy.special import factorial, genlaguerre
from scipy.constants import pi


def eigenfunction_2d_equation17(x, y, n, m):
    """Evaluate the phase-convention eigenfunction psi_{n,m}(x, y).

    This is the function described in manuscript "Energy level
    splitting for weakly interacting bosons in a harmonic trap" as
    equation (17).

    Implements the Laguerre-based radial eigenfunction with angular dependence
    e^{i m phi} and the normalization convention used in the manuscript
    (see eqs. (13)-(17)). Cartesian inputs are converted to polar
    coordinates internally.

    Args:
        x (float or ndarray): x coordinate(s).
        y (float or ndarray): y coordinate(s).
        n (int): Principal quantum number. Must satisfy n >= |m| and n-m even.
        m (int): Angular momentum quantum number (m >= 0 for this form).

    Returns:
        complex or ndarray: The value(s) of psi_{n,m}(x, y). If the supplied
            quantum numbers are invalid (e.g. n < m or n-m odd) returns 0.0.
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # Validation: n >= m >= 0, n - m even (as per eq 17 form)
    if m < 0 or n < m or (n - m) % 2 != 0:
        return 0.0

    # Factorials in denominator: (1/2(n-m))! and (1/2(n+m))!
    fact1 = factorial(int((n - m) / 2))
    fact2 = factorial(int((n + m) / 2))

    # Normalization factor from eq (17)
    norm_factor = np.sqrt(fact1 / fact2)

    # Laguerre polynomial L^m_{(n-m)/2}(r^2)
    laguerre_arg = r**2
    L = genlaguerre(m, int((n - m) / 2))(laguerre_arg)

    # Full wavefunction per eq (17)
    psi = (norm_factor * (r**m / np.sqrt(pi)) * L *
           np.exp(-r**2 / 2) * np.exp(1j * m * phi))

    return psi


# Step 1: Discretize space and build single-particle orbitals
def build_single_particle_orbitals(positions, quantum_numbers=None):
    """Build an orbital matrix by evaluating single-particle eigenfunctions.

    The returned matrix has shape `(n_orbitals, n_modes)` and its element
    `(i, j)` is the value of orbital `i` evaluated at spatial position `j`.

    Args:
        positions (ndarray): Array of shape `(n_modes, 2)` containing (x, y)
            coordinates for each spatial mode chi_j.
        quantum_numbers (list[tuple] or None): Sequence of `(n, m)` quantum
            numbers defining which eigenfunctions to include. If ``None``, a
            small default set is used.

    Returns:
        ndarray: Complex array `orbital_matrix` with shape
            `(n_orbitals, n_modes)` where `orbital_matrix[i, j] = psi_i(chi_j)`.
    """
    # Example: label orbitals by (n, m) along some sequence
    # You can change this to match the spectrum you want.
    if quantum_numbers is None:
        quantum_numbers = [(0, 0), (1, 1), (2, 2)]

    n_orbitals = len(quantum_numbers)
    n_modes = positions.shape[0]
    orbital_matrix = np.zeros((n_orbitals, n_modes), dtype=complex)

    for i, (n_qn, m_qn) in enumerate(quantum_numbers):
        for j, (x, y) in enumerate(positions):
            orbital_matrix[i, j] = eigenfunction_2d_equation17(x, y, n_qn, m_qn)
    return orbital_matrix

# Step 2: Orthogonalization (Löwdin / symmetric)
# After evaluating single-particle orbitals on a discrete set of detector
# positions we obtain an (n_orbitals x m) orbital matrix. To use these orbitals
# as input to a linear-optical interferometer we require an orthonormal set of
# mode functions. Löwdin (symmetric) orthogonalization produces the closest
# orthonormal set to the original rows while preserving the subspace they span.
def lowdin(P):
    """Compute the Löwdin (symmetric) orthogonalization of a matrix.

    Args:
        P (ndarray): Input array with shape (n, k). Rows are the vectors to
            orthogonalize.

    Returns:
        ndarray: Matrix `B` with same shape as `P` whose rows are orthonormal
            under the Hermitian inner product (so that `B @ B.conj().T == I`
            or `B.T.conj() @ B == I` depending on shapes).

    Raises:
        AssertionError: If internal numerical checks for orthonormality fail.
    """
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

# Step 3: Build a Piquasso boson-sampling program
# We convert a unitary describing a linear interferometer into a Piquasso
# program. The program prepares a vacuum, creates Fock excitations according to
# an input occupation vector, applies the interferometer and finally measures all
# modes in the photon-number basis.

def build_piquasso_program(unitary, input_occupation):
    """Construct a Piquasso `Program` implementing a linear interferometer.

    The constructed program prepares the vacuum state across `m` modes, adds
    Fock excitations according to `input_occupation`, applies the linear
    interferometer specified by `unitary` and appends a particle-number
    measurement on all modes.

    Args:
        unitary (ndarray): Square complex matrix of shape (m, m) describing a
            linear interferometer acting on `m` modes.
        input_occupation (Sequence[int]): Length-`m` iterable with the number
            of photons to create in each input mode.

    Returns:
        pq.Program: A Piquasso program ready for execution by a sampler.
    """
    m = unitary.shape[0]
    instructions = []

    instructions.append(pq.Vacuum().on_modes(*range(m)))

    # Prepare Fock input state: add photons to specified modes
    for mode, photon_count in enumerate(input_occupation):
        if photon_count == 1:
            instructions.append(pq.Create().on_modes(mode))

    # Apply linear interferometer corresponding to unitary
    instructions.append(pq.Interferometer(unitary))

    # Measure all modes in photon number basis
    instructions.append(pq.ParticleNumberMeasurement())

    return pq.Program(instructions=instructions)

# Step 4: Map detection patterns back to particle positions
# After sampling photon-number outputs from the boson sampler, we map detector
# clicks (occupied modes) back to spatial coordinates. For collision-free
# samples (0/1 per mode) this yields a configuration of particle positions that
# can be used to evaluate the classical observable V(X).

def pattern_to_positions(pattern, positions):
    """Map a detected photon-number pattern to particle positions.

    For collision-free samples (0 or 1 photons per mode) this returns the
    list of spatial coordinates corresponding to modes that registered a
    photon. If collisions (multiple photons in a mode) are present the
    function returns the positions of occupied modes with multiplicity omitted.

    Args:
        pattern (Sequence[int]): Photon counts per mode (length `m`).
        positions (ndarray): Array of shape (m, 2) giving (x, y) coordinates
            for each mode.

    Returns:
        ndarray: Array of shape (n_bosons, 2) containing positions of each
            detected boson (collision-free assumption recommended).
    """
    idx = [i for i, n in enumerate(pattern) if n == 1]
    return positions[idx, :]

# Step 5: Define the target perturbation V(X)
# In the paper the authors evaluate a classical perturbation V(X).
# Here we use a simple Efimov-inspired 3-body potential as an illustrative
# example. This function computes the classical observable h(X) = V(X).

def efimov_potential_3body(positions_xyz, C=1.0):
    """Evaluate an Efimov-inspired 3-body potential for three particles.

    This is an illustrative potential inspired by Efimov-like scaling used in
    the manuscript. The function accepts an array of three 2D coordinates and
    returns a scalar potential value. A small-distance cutoff is handled by
    returning 0.0 when the metric R^2 would be zero.

    Args:
        positions_xyz (ndarray): Array of shape (3, 2) containing (x, y)
            coordinates for three particles.
        C (float): Strength parameter for the potential (default: 1.0).

    Returns:
        float: The scalar potential value V(X) for the input configuration.
    """
    x = positions_xyz
    r12 = np.linalg.norm(x[0] - x[1])
    r13 = np.linalg.norm(x[0] - x[2])
    r23 = np.linalg.norm(x[1] - x[2])

    R2 = (2.0 / 3.0) * (r12**2 + r13**2 + r23**2)
    if R2 == 0:
        return 0.0  # Hard-shell cutoff in discrete model
    return -(C + 1.0 / 4.0) / R2

def extend_to_orthonormal_rows(mat_with_ortogonal_rows, seed=None, tol=1e-12,
                                 max_tries=10000):
    """Extend a set of mutually orthogonal rows to a full orthonormal basis.

    Given an input matrix whose rows are mutually orthogonal (but not a full
    basis), this routine samples random complex vectors, removes components
    along the existing rows (Hermitian Gram-Schmidt style) and accepts
    independent vectors until a complete set of `n` rows is obtained. Rows are
    then normalized so that the returned matrix `Q` satisfies
    `Q @ Q.conj().T == I`.

    Args:
        mat_with_ortogonal_rows (ndarray): Array of shape (r, n) where `r < n`
            and rows are mutually orthogonal under the Hermitian inner product.
        seed (int or None): Optional RNG seed for reproducibility (default: None).
        tol (float): Numerical tolerance for independence checks (default: 1e-12).
        max_tries (int): Maximum number of random trials used to find
            additional independent vectors (default: 10000).

    Returns:
        ndarray: A square `(n, n)` complex matrix whose rows are orthonormal.

    Raises:
        RuntimeError: If the algorithm fails to find enough independent
            vectors within `max_tries` attempts.
    """
    rng = np.random.default_rng(seed)

    basis = [mat_with_ortogonal_rows[i].copy()
             for i in range(len(mat_with_ortogonal_rows))]

    def hermitian_proj_coeff(b, w):
        """Compute Hermitian projection coefficient.

        Coefficient c such that w - c*b removes component along b:
        c = <b,w>/<b,b> with Hermitian inner product.
        """
        bb = np.vdot(b, b)
        if np.abs(bb) <= tol:
            return 0.0j
        return np.vdot(b, w) / bb

    def orthogonalize(v, basis_list):
        """Orthogonalize vector v against basis_list using Gram-Schmidt.

        Args:
            v (ndarray): Vector to orthogonalize.
            basis_list (list): List of orthogonal vectors.

        Returns:
            ndarray: Orthogonalized vector.
        """
        w = v.copy()
        for b in basis_list:
            w = w - hermitian_proj_coeff(b, w) * b
        return w

    n_modes = mat_with_ortogonal_rows.shape[1]
    tries = 0
    while len(basis) < n_modes:
        if tries >= max_tries:
            msg = ("Could not generate enough independent vectors; "
                   "increase max_tries or check input.")
            raise RuntimeError(msg)
        tries += 1

        # Generate random complex vector (real + i*imag)
        v = rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)
        w = orthogonalize(v, basis)

        if np.linalg.norm(w) > tol:
            basis.append(w)

    Q = np.stack(basis, axis=0)  # (n_modes, n_modes)

    def per_row_norms(matrix):
        """Compute Hermitian norm for each row.

        Args:
            matrix (ndarray): Input matrix.

        Returns:
            ndarray: Norm of each row.
        """
        return np.sqrt(np.real(np.einsum('ij,ij->i', matrix.conj(), matrix)))

    # Normalize rows using Hermitian norm ||q|| = sqrt(<q,q>)
    norms = per_row_norms(Q)
    Q = Q / norms[:, None]

    assert np.allclose(per_row_norms(Q), 1)
    return Q

def perm_3x3(A):
    """Compute the permanent of a 3x3 matrix.

    The permanent is similar to the determinant but without sign changes.

    Args:
        A (ndarray): 3x3 complex matrix.

    Returns:
        complex: The permanent of the matrix.
    """
    return (
        A[0, 0]*A[1, 1]*A[2, 2] +
        A[0, 0]*A[2, 1]*A[1, 2] +
        A[1, 0]*A[0, 1]*A[2, 2] +
        A[1, 0]*A[2, 1]*A[0, 2] +
        A[2, 0]*A[0, 1]*A[1, 2] +
        A[2, 0]*A[1, 1]*A[0, 2]
    )

def get_classical_vals(input_occupation, unitary, positions, n_samples=10000,
                        potential_fn=efimov_potential_3body):
    """Get classical Monte Carlo estimates from theoretical distribution.

    Enumerate all collision-free outputs (size-3 subsets of output modes) and
    compute their probabilities from the unitary matrix, then sample from the
    resulting distribution.

    Args:
        input_occupation (Sequence[int]): Photon occupation vector.
        unitary (ndarray): Square unitary matrix of shape (m, m).
        positions (ndarray): Array of shape (m, 2) with mode positions.
        n_samples (int): Number of samples to draw (default: 10000).
        potential_fn (callable): Function to evaluate on sampled positions.

    Returns:
        tuple: (mean, standard_error) of the potential function values.
    """
    # Enumerate all collision-free outputs (size-3 subsets of output modes)
    m = positions.shape[0]

    in_modes = [i for i, n in enumerate(input_occupation) if n == 1]

    # Row-reduction: extract rows corresponding to input modes
    Uin = unitary[:, np.array(in_modes)]  # m x 3

    # Form all possible output mode combinations
    patterns = list(itertools.combinations(range(m), 3))
    weights = np.empty(len(patterns), dtype=float)

    for k, S in enumerate(patterns):
        sub = Uin[S, :]  # 3 x 3 submatrix
        weights[k] = np.abs(perm_3x3(sub))**2

    # Condition on collision-free outputs
    Z = weights.sum()
    pmf = weights / Z

    assert pmf.sum() - 1.0 < 1e-10, "Probability distribution not normalized"

    rng = np.random.default_rng(42)
    idx = rng.choice(len(patterns), size=n_samples, p=pmf, replace=True)

    vals = []
    for i in idx:
        S = patterns[i]
        X = positions[np.array(S), :]  # Always 3 points
        if len(X) != 3:
            continue
        vals.append(potential_fn(X))

    vals = np.asarray(vals, dtype=float)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))

def get_bs_vals(input_occupation, unitary, positions, n_samples=10000,
                potential_fn=efimov_potential_3body):
    """Sample boson sampler and accumulate potential function values.

    Args:
        input_occupation (Sequence[int]): Photon occupation vector.
        unitary (ndarray): Square unitary matrix of shape (m, m).
        positions (ndarray): Array of shape (m, 2) with mode positions.
        n_samples (int): Number of samples to attempt (default: 10000).
        potential_fn (callable): Function to evaluate on positions.

    Returns:
        tuple: (mean, standard_error, n_valid_samples) of valid 3-photon events.

    Raises:
        RuntimeError: If no valid 3-photon samples are collected.
    """
    values = []
    m = positions.shape[0]
    sim = pq.SamplingSimulator(d=m)

    # Build and execute Piquasso program
    prog = build_piquasso_program(unitary, input_occupation)
    result = sim.execute(prog, shots=n_samples)

    for pattern in result.samples:
        idx = [i for i, n in enumerate(pattern) if n == 1]
        X = positions[idx, :]

        # Discard samples that are not collision-free three-click events
        if len(X) != 3:
            continue
        values.append(potential_fn(X))

    if len(values) == 0:
        raise RuntimeError("No valid 3-photon samples collected.")
    return np.mean(values), np.std(values) / np.sqrt(len(values)), len(values)


# Step 6: Boson-sampling-assisted Monte Carlo integration
# This step runs the sampler many times to generate detection patterns X drawn
# approximately from g(X) ≈ |Ψ₀(X)|². Collision-free events are mapped back to
# particle positions and the classical observable h(X) = V(X) is evaluated for
# each sample. The Monte Carlo estimator is the empirical mean of h(X) over the
# collected valid samples.
def boson_sampling_monte_carlo(
    positions,
    input_occupation=None,
    n_samples=1000,
    potential_fn=efimov_potential_3body,
):
    """Estimate an expectation value by boson-sampling-assisted Monte Carlo.

    The procedure implemented here follows the algorithm described in the
    manuscript:

    1. Construct single-particle orbitals evaluated at detector/mode positions
       to form an orbital matrix.
    2. Apply Löwdin symmetric orthogonalization to obtain orthonormal rows
       spanning the same subspace.
    3. Extend the orthonormal rows to a full unitary used as the linear
       interferometer in a boson sampler.
    4. Use Piquasso to sample collision-free detection events X distributed
       according to g(X) ≈ |Ψ₀(X)|² and evaluate h(X) = V(X) classically for
       each sample.

    Args:
        positions (ndarray): Array of shape (m, 2) with coordinates for `m`
            spatial modes.
        input_occupation (Sequence[int] or None): Length-`m` photon occupation
            vector describing the Fock input state. If `None`, defaults to one
            photon in the first three modes (default: None).
        n_samples (int): Number of sampler executions to attempt (default: 1000).
        potential_fn (callable): Function V(X) that maps particle configuration
            X (array of positions) to a scalar value
            (default: efimov_potential_3body).

    Returns:
        tuple: `((bs_mean, bs_error), (classical_mean, classical_error))` where
            the first pair is from boson sampling and the second is the
            theoretical distribution.
    """
    m = positions.shape[0]
    if input_occupation is None:
        # Default: 1 photon in the first 3 modes
        input_occupation = [1, 1, 1] + [0] * (m - 3)

    # Step 1: Construct orbitals and unitary
    orbital_matrix = build_single_particle_orbitals(positions)
    assert np.linalg.matrix_rank(orbital_matrix) == orbital_matrix.shape[0], \
        "Orbital matrix should have full row rank"

    # Step 2: Apply Löwdin orthogonalization
    orthogonal_rows_mat = lowdin(orbital_matrix)
    assert np.linalg.matrix_rank(orthogonal_rows_mat) == orbital_matrix.shape[0], \
        "Orthogonal rows should have full row rank"

    # Step 3: Extend to full unitary
    unitary = extend_to_orthonormal_rows(orthogonal_rows_mat, seed=0)

    # Step 4: Sample from boson sampler
    bs_mean, bs_error, len_bs_vals = get_bs_vals(
        input_occupation, unitary, positions, n_samples, potential_fn)

    # Step 5: Get theoretical distribution estimate
    classical_mean, classical_error = get_classical_vals(
        input_occupation, unitary, positions, len_bs_vals, potential_fn)

    return (bs_mean, bs_error), (classical_mean, classical_error)

# Step 7: Example usage
if __name__ == "__main__":
    # Discretize a line into 12 modes on x-axis
    n_modes = 12
    xs = np.linspace(-3.0, 3.0, n_modes)
    ys = np.zeros_like(xs)
    positions = np.stack([xs, ys], axis=1)
    n_samples = 10000

    (bs_estimate, bs_error), (classical_estimate, classical_error) = (
        boson_sampling_monte_carlo(positions, n_samples=n_samples))

    print(f"Piquasso estimated E^(1):       {bs_estimate:.4f} +/- {bs_error:.4f}")
    print(f"Naive permanent computation E^(1): {classical_estimate:.6f} +/- "
          f"{classical_error:.6f}")
    print(f"Difference: {abs(bs_estimate - classical_estimate):.6f}")
