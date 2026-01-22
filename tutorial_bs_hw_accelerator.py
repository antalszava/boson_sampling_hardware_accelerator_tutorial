from ast import pattern
from random import seed
import numpy as np

import piquasso as pq
from numpy.linalg import svd

import numpy as np
from scipy.special import factorial, genlaguerre
from scipy.constants import pi
from scipy.linalg import sqrtm

import numpy as np
import itertools

def perm_3x3(A):
    return (
        A[0, 0]*A[1, 1]*A[2, 2] +
        A[0, 0]*A[2, 1]*A[1, 2] +
        A[1, 0]*A[0, 1]*A[2, 2] +
        A[1, 0]*A[2, 1]*A[0, 2] +
        A[2, 0]*A[0, 1]*A[1, 2] +
        A[2, 0]*A[1, 1]*A[0, 2]
    )

"""
Boson-sampling Monte Carlo accelerator tutorial

This script accompanies the manuscript "Experimental demonstration of boson
sampling as a hardware accelerator for Monte Carlo integration" and provides a
a Piquasso-based script that implements the core algorithm.

The workflow below follows these high-level steps:

1. Discretize space and build single-particle orbitals evaluated at detector/mode
    positions (orbital matrix).
2. Orthogonalize the orbital matrix using Löwdin (symmetric) orthogonalization
    to obtain a matrix with orthonormal rows.
3. Extend the orthogonal rows into a full unitary (interferometer) that can be
    executed on a linear optical sampler (here we use a random completion).
4. Use Piquasso to build and execute a boson-sampling program implementing the
 unitary and sample output photon patterns.
5. Map detected photon patterns back to particle positions and evaluate the
    target potential function (e.g. Efimov-inspired 3-body potential) to form
    Monte Carlo estimates.
"""

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
    fact1 = factorial((n - m)/2)
    fact2 = factorial((n + m)/2)

    # Normalization factor from eq (17)
    norm_factor = np.sqrt(fact1 / fact2)

    # Laguerre polynomial L^m_{(n-m)/2}(r^2) - no absolute value on m
    laguerre_arg = r**2
    L = genlaguerre(m, (n - m)/2)(laguerre_arg)

    # Full wavefunction per eq (17)
    psi = norm_factor * (r**m / np.sqrt(pi)) * L * np.exp(-r**2 / 2) * np.exp(1j * m * phi)

    return psi


# Step 1 - Discretize space and build single-particle orbitals
"""
 Discretize a spatial region into a set of `m` detector/mode positions and
evaluate single-particle eigenfunctions psi_{n,m}(x,y) on those positions. The
result is an orbital matrix with shape (n_orbitals, n_modes) which encodes the
overlap between spatial modes and basis orbitals. This matrix is the starting
    point for constructing a unitary interferometer (via Löwdin orthogonalization).
"""

# ----- 1. Discretize space and build orbitals -----
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

    for i, (n, m) in enumerate(quantum_numbers):
        for j, (x, y) in enumerate(positions):
            orbital_matrix[i, j] = eigenfunction_2d_equation17(x, y, n, m)
    return orbital_matrix

"""
Step 2 - Orthogonalization (Löwdin / symmetric)

After evaluating single-particle orbitals on a discrete set of detector
positions we obtain an (n_orbitals x m) orbital matrix. To use these orbitals
as input to a linear-optical interferometer we require an orthonormal set of
mode functions. Löwdin (symmetric) orthogonalization produces the closest
orthonormal set to the original rows while preserving the subspace they span.
We implement this with an SVD-based construction which is numerically stable
and convenient for our purpose of constructing a boson-sampling program.
"""

"""
Step 3 - Build a Piquasso boson-sampling program

Description:
We convert a unitary describing a linear interferometer into a Piquasso
program. The program prepares a vacuum, creates Fock excitations according to
an input occupation vector, applies the interferometer and finally measures all
modes in the photon-number basis. This program can then be executed by one of
Piquasso's simulators or an experimental backend.
"""

# ----- 3. Piquasso boson-sampling circuit -----

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

    # Prepare Fock input state
    for mode, n in enumerate(input_occupation):
        if n == 1:
            instructions.append(pq.Create().on_modes(mode))

    # Apply linear interferometer corresponding to unitary
    instructions.append(pq.Interferometer(unitary))

    # Measure all modes in photon number basis
    instructions.append(pq.ParticleNumberMeasurement())

    return pq.Program(instructions=instructions)

"""
Step 4 - Map detection patterns back to particle positions

After sampling photon-number outputs from the boson sampler, we map detector
clicks (occupied modes) back to spatial coordinates. For collision-free
samples (0/1 per mode) this yields a configuration of particle positions that
can be used to evaluate the classical observable V(X).
"""

# ----- 4. Map detection pattern to particle positions -----

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

"""
Step 5 - Define the target perturbation V(X)

In the paper the authors evaluate a classical perturbation V(X) (here we use a
simple Efimov-inspired 3-body potential as an illustrative example). After
mapping detection patterns to particle positions, this function computes the
classical observable h(X) = V(X) for use in the Monte Carlo estimator.
"""

# ----- 5. Define perturbation V(X) (Efimov-like example) -----

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
        return 0.0  # hard-shell cutoff in discrete model
    return -(C + 1/4) / R2

def extend_to_orthonormal_rows(mat_with_ortogonal_rows, seed=None, tol=1e-12, max_tries=10_000):
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
        seed (int or None): Optional RNG seed for reproducibility.
        tol (float): Numerical tolerance for independence checks.
        max_tries (int): Maximum number of random trials used to find
            additional independent vectors.

    Returns:
        ndarray: A square `(n, n)` complex matrix whose rows are orthonormal.

    Raises:
        RuntimeError: If the algorithm fails to find enough independent
            vectors within `max_tries` attempts.
    """
    rng = np.random.default_rng(seed)

    basis = [mat_with_ortogonal_rows[i].copy() for i in range(len(mat_with_ortogonal_rows))]

    def hermitian_proj_coeff(b, w):
        # coefficient c such that w - c*b removes component along b:
        # c = <b,w>/<b,b> with Hermitian inner product implemented by vdot (conjugates first arg)
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

    def per_row_norms(matrix):
        return np.sqrt(np.real(np.einsum('ij,ij->i', matrix.conj(), matrix)))

    # Better: per-row norms using the Hermitian norm ||q|| = sqrt(<q,q>)
    norms = per_row_norms(Q)
    Q = Q / norms[:, None]

    assert np.allclose(per_row_norms(Q), 1)
    return Q

def get_classical_vals(input_occupation, unitary, positions, n_samples=10000, potential_fn=efimov_potential_3body):
    # Enumerate all collision-free outputs S (size-3 subsets of output modes)

    m = positions.shape[0]

    in_modes = [i for i, n in enumerate(input_occupation) if n == 1]

    # Row-reduction
    Uin = unitary[np.array(in_modes), :]  # 3 x m

    # Choose what modes one photon can be detected - form the combinations
    patterns = list(itertools.combinations(range(m), 3))
    weights = np.empty(len(patterns), dtype=float)

    for k, S in enumerate(patterns):
        sub = Uin[:, S]       # 3 x 3
        weights[k] = np.abs(perm_3x3(sub))**2

    # Condition on collision-free outputs (since your quantum loop discards others)
    Z = weights.sum()
    pmf = weights / Z

    assert pmf.sum() - 1.0 < 1e-10

    rng = 0
    rng = np.random.default_rng(rng)
    idx = rng.choice(len(patterns), size=n_samples, p=pmf, replace=True)

    vals = []
    for i in idx:
        S = patterns[i]
        X = positions[np.array(S), :]  # always 3 points
        vals.append(potential_fn(X))

    vals = np.asarray(vals, dtype=float)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))

def get_bs_positions(input_occupation, unitary, positions):

    sim = pq.SamplingSimulator(d=12, config=pq.Config(seed_sequence=0))

    # 2) Build Piquasso program and simulator
    prog = build_piquasso_program(unitary, input_occupation)
    result = sim.execute(prog)
    pattern = result.samples  # list of photon counts per mode
    idx = [i for i, n in enumerate(pattern[0]) if n == 1]
    return positions[idx, :]

def get_bs_vals(input_occupation, unitary, positions, n_samples=10000, potential_fn=efimov_potential_3body):

    # 3) Sample patterns and accumulate h(X)
    values = []
    for _ in range(n_samples):
        X = get_bs_positions(input_occupation, unitary, positions)

        # Discard samples that are not collision-free three-click events
        if len(X) != 3:
            continue
        values.append(potential_fn(X))

    if len(values) == 0:
        raise RuntimeError("No valid 3-photon samples collected.")
    return np.mean(values), np.std(values) / np.sqrt(len(values)), len(values)


# ----- 6. Boson-sampling-assisted Monte Carlo integration -----
"""
Step 6 - Monte Carlo sampling loop and estimator

This step runs the sampler many times to generate detection patterns X drawn
approximately from g(X) approx |Psi_0(X)|^2. Collision-free events are mapped back to
particle positions and the classical observable h(X) = V(X) is evaluated for
each sample. The Monte Carlo estimator is the empirical mean of h(X) over the
collected, valid samples; the function returns the mean and the standard error
of the mean. In practice one should monitor the fraction of valid (collision-
free) samples and increase `n_samples` or adjust the encoding to obtain enough
statistics.
"""
def boson_sampling_monte_carlo(
    positions,
    input_occupation=None,
    n_samples=1000,
    potential_fn=efimov_potential_3body,
):
    """Estimate an expectation value by boson-sampling-assisted Monte Carlo.

    The procedure implemented here follows the algorithm described in the
    manuscript:

     1. Construct single-particle orbitals evaluated at detector/mode
         positions to form an orbital matrix `chi`.
     2. Apply Löwdin symmetric orthogonalization to obtain orthonormal rows
         spanning the same subspace as `chi`.
    3. Extend the orthonormal rows to a full unitary `U` which is used as the
       linear interferometer in a boson sampler.
    4. Use Piquasso to sample collision-free detection events X distributed
    according to g(X) approx |Psi_0(X)|^2 and evaluate h(X) = V(X) classically for
       each sample. The Monte Carlo estimate is the empirical mean of h(X).

    Args:
        positions (ndarray): Array of shape (m, 2) with coordinates for `m`
            spatial modes.
        input_occupation (Sequence[int] or None): Length-`m` photon occupation
            vector describing the Fock input state. If `None`, a default state
            with one photon in the first three modes is used.
        n_samples (int): Number of sampler executions to attempt.
        potential_fn (callable): Function V(X) that maps a particle configuration
            `X` (array of positions) to a scalar value.
        simulator_cls (type): Piquasso simulator class used to execute the
            program (defaults to `pq.SamplingSimulator`).

    Returns:
        tuple: `(estimate, stderr)` where `estimate` is the sample mean of
            `h(X)` and `stderr` is the standard error of the mean.
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

    bs_vals1, bs_vals2, len_bs_vals = get_bs_vals(input_occupation, unitary, positions, n_samples, potential_fn)
    classical_vals = get_classical_vals(input_occupation, unitary, positions, len_bs_vals, potential_fn)
    return (bs_vals1, bs_vals2), classical_vals

# ----- 7. Example usage -----
if __name__ == "__main__":
    # Discretize a line or 2D strip into modes
    # Example: 12 modes on x-axis, y = 0
    n_modes = 12
    xs = np.linspace(-3.0, 3.0, n_modes)
    ys = np.zeros_like(xs)
    positions = np.stack([xs, ys], axis=1)

    (estimate1, error1), (estimate2, error2) = boson_sampling_monte_carlo(
        positions,
        n_samples=1000,
    )
    print(f"Estimated E^(1) approx {estimate1:.4f} +/- {error1:.4f}")
    print(f"Classical estimated E^(1) approx {estimate2:.6f} +/- {error2:.6f}")
