"""
VQE optimization module.

Classical optimization loop implementation for the Variational Quantum Eigensolver algorithm.
"""

import numpy as np
from functools import partial
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


def objective(
    params: np.ndarray,
    iteration_count: list,
    energy_history: list,
    ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    energy_func
) -> float:
    """
    Objective function for VQE optimization.
    
    This function wraps the energy evaluation and tracks optimization progress.
    It's called by the classical optimizer at each iteration with new parameter values.
    
    Parameters
    ----------
    params : np.ndarray
        Current parameter values being evaluated
    iteration_count : list
        Mutable list containing iteration counter [current_iteration]
    energy_history : list
        Mutable list to store energy values at each iteration
    ansatz : QuantumCircuit
        The parameterized quantum circuit
    hamiltonian : SparsePauliOp
        The Hamiltonian operator
    energy_func : callable
        Energy evaluation function with signature:
        energy_func(circuit, hamiltonian, params) -> float
        
    Returns
    -------
    float
        The energy value to be minimized
        
    Notes
    -----
    - Increments iteration counter
    - Appends energy to history for analysis
    - Prints progress every 20 iterations
    - Thread-safe for sequential optimization
    """
    # Increment iteration counter
    iteration_count[0] += 1
    
    # Evaluate energy with current parameters
    energy = energy_func(ansatz, hamiltonian, params)
    
    # Store energy in history
    energy_history.append(energy)
    
    # Print progress periodically
    if iteration_count[0] % 20 == 0 or iteration_count[0] == 1:
        print(f"Iteration {iteration_count[0]:3d}: E = {energy:+.6f}")
    
    return energy


def run_vqe(
    ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    energy_func,
    initial_params: np.ndarray = None,
    method: str = "COBYLA",
    max_iter: int = 200
) -> dict:
    """
    Run the Variational Quantum Eigensolver (VQE) algorithm.
    
    VQE is a hybrid quantum-classical algorithm that:
    1. Prepares a parameterized quantum state |ψ(θ)⟩ on quantum hardware
    2. Measures energy E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
    3. Uses classical optimization to find parameters θ* that minimize E(θ)
    
    By the variational principle, E(θ) ≥ E₀ (ground state energy),
    so minimizing over θ approximates the ground state within the
    expressivity of the ansatz.
    
    Parameters
    ----------
    ansatz : QuantumCircuit
        Parameterized quantum circuit (the trial state generator).
        Should have num_parameters matching the optimization dimension.
    hamiltonian : SparsePauliOp
        The Hamiltonian whose ground state we seek.
        Must be Hermitian (real eigenvalues).
    energy_func : callable
        Energy evaluation function with signature:
        energy_func(circuit, hamiltonian, params) -> float
        
        This abstraction allows seamless swapping between:
        - Exact statevector simulation (validation)
        - Shot-based simulation (realistic noise)
        - Real quantum hardware execution
    initial_params : np.ndarray, optional
        Starting parameter values. If None, initializes randomly
        in [0, 2π) which is appropriate for rotation gates.
    method : str, default="COBYLA"
        Classical optimizer to use. Options include:
        - "COBYLA": Derivative-free, robust to noise (default)
        - "SLSQP": Sequential least squares, faster for smooth landscapes
        - "L-BFGS-B": Limited-memory BFGS, efficient for many parameters
        - "Powell": Derivative-free, conjugate directions
    max_iter : int, default=200
        Maximum number of optimizer iterations.
        Each iteration evaluates energy once (or more for some optimizers).
        
    Returns
    -------
    dict
        Result dictionary containing:
        - 'optimal_params' : np.ndarray
            Best parameter values found
        - 'optimal_energy' : float
            Minimum energy achieved (ground state estimate)
        - 'num_iterations' : int
            Total function evaluations performed
        - 'success' : bool
            Whether optimization converged successfully
        - 'energy_history' : list[float]
            Energy values at each iteration (for analysis/plotting)
            
    Examples
    --------
    >>> from scripts.ansatz import he_ansatz_2q
    >>> from scripts.hamiltonian import heisenberg_xxz_2q
    >>> from scripts.energy_evaluation import energy_expectation
    >>> 
    >>> # Set up problem
    >>> ansatz, theta = he_ansatz_2q(depth=1)
    >>> H = heisenberg_xxz_2q(J=1.0, delta=1.0)
    >>> 
    >>> # Run VQE with exact statevector evaluation
    >>> result = run_vqe(
    ...     ansatz=ansatz,
    ...     hamiltonian=H,
    ...     energy_func=energy_expectation,
    ...     method="COBYLA",
    ...     max_iter=100
    ... )
    >>> 
    >>> print(f"Ground state energy: {result['optimal_energy']:.6f}")
    >>> print(f"Converged: {result['success']}")
    
    >>> # Run VQE with shot-based simulation
    >>> from scripts.energy_evaluation import energy_expectation_shots
    >>> result_shots = run_vqe(
    ...     ansatz=ansatz,
    ...     hamiltonian=H,
    ...     energy_func=lambda c, h, p: energy_expectation_shots(c, h, p, shots=2000),
    ...     initial_params=result['optimal_params'],  # Warm start
    ...     max_iter=50
    ... )
    
    Notes
    -----
    Design principles:
    - Flexible energy_func parameter enables easy backend switching
    - functools.partial binds extra arguments for scipy.optimize interface
    - Tracks full optimization history for analysis
    - Prints progress for long-running optimizations
    
    Optimizer choice:
    - COBYLA: Best for noisy objectives (shot-based, hardware)
    - SLSQP/L-BFGS-B: Faster for smooth objectives (statevector)
    - For hardware: Consider SPSA (simultaneous perturbation stochastic approximation)
    
    Initialization strategies:
    - Random: Simple, unbiased (default)
    - Warm start: Use simulation result for hardware runs
    - Heuristic: Problem-specific good guesses
    
    Convergence considerations:
    - More iterations: Better convergence, higher cost
    - Shot noise: May prevent exact convergence
    - Local minima: Try multiple random starts
    - Barren plateaus: Increase ansatz depth or use problem-inspired ansatz
    """
    num_params = ansatz.num_parameters
    
    # Initialize parameters if not provided
    if initial_params is None:
        # Random initialization in [0, 2π) for rotation gates
        initial_params = np.random.uniform(0, 2 * np.pi, size=num_params)
    
    # Validate initial parameters
    if len(initial_params) != num_params:
        raise ValueError(
            f"initial_params length ({len(initial_params)}) does not match "
            f"ansatz parameters ({num_params})"
        )
    
    # Track optimization progress
    iteration_count = [0]  # Mutable list for closure
    energy_history = []    # Store all energies
    
    # Create partial function that binds extra arguments
    # scipy.optimize.minimize expects f(x) -> float signature
    objective_func = partial(
        objective,
        iteration_count=iteration_count,
        energy_history=energy_history,
        ansatz=ansatz,
        hamiltonian=hamiltonian,
        energy_func=energy_func
    )
    
    # Print VQE setup information
    print("Starting VQE optimization...")
    print(f"Optimizer: {method}, Max iterations: {max_iter}")
    print(f"Ansatz: {num_params} parameters, depth={ansatz.depth()}")
    print(f"Initial energy: {objective_func(initial_params):+.6f}")
    print("-" * 50)
    
    # Run classical optimization
    opt_result = minimize(
        objective_func,
        initial_params,
        method=method,
        options={'maxiter': max_iter}
    )
    
    # Print completion summary
    print("-" * 50)
    print(f"Optimization complete!")
    print(f"Final energy: {opt_result.fun:+.6f}")
    print(f"Total iterations: {iteration_count[0]}")
    print(f"Success: {opt_result.success}")
    if not opt_result.success:
        print(f"Message: {opt_result.message}")
    
    # Return comprehensive results
    return {
        'optimal_params': opt_result.x,
        'optimal_energy': opt_result.fun,
        'num_iterations': iteration_count[0],
        'success': opt_result.success,
        'energy_history': energy_history,
        'optimizer_message': opt_result.message
    }
