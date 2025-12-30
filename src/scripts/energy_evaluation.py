"""
Energy evaluation module.

Functions for evaluating energy expectation values using different backends:
exact statevector simulation, shot-based simulation, and real quantum hardware.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator


def energy_expectation(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    param_values: np.ndarray
) -> float:
    """
    Compute energy expectation value using exact statevector simulation.
    
    This function evaluates ⟨ψ(θ)|H|ψ(θ)⟩ exactly without measurement noise.
    It's ideal for validation and small systems but doesn't scale to large
    qubit counts due to exponential memory requirements (2^n amplitudes).
    
    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized quantum circuit (the ansatz)
    hamiltonian : SparsePauliOp
        The Hamiltonian operator
    param_values : np.ndarray
        Parameter values to bind to the circuit
        
    Returns
    -------
    float
        The exact energy expectation value (real part only)
        
    Examples
    --------
    >>> from scripts.ansatz import he_ansatz_2q
    >>> from scripts.hamiltonian import heisenberg_xxz_2q
    >>> 
    >>> ansatz, theta = he_ansatz_2q(depth=1)
    >>> H = heisenberg_xxz_2q(J=1.0, delta=1.0)
    >>> params = np.random.uniform(0, 2*np.pi, size=4)
    >>> energy = energy_expectation(ansatz, H, params)
    
    Notes
    -----
    - Uses StatevectorEstimator for exact (noiseless) evaluation
    - No sampling error or hardware noise
    - Only suitable for small systems (< ~20 qubits)
    - Returns real part of energy (Hamiltonians are Hermitian)
    """
    estimator = StatevectorEstimator()
    
    # Bind parameter values to get concrete circuit
    bound = circuit.assign_parameters(param_values, inplace=False)
    
    # Run estimator to compute expectation value
    job = estimator.run([(bound, hamiltonian)])
    result = job.result()[0]
    
    # Extract energy (scalar value, already real for Hermitian operators)
    return float(np.real(result.data.evs))


def energy_expectation_shots(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    param_values: np.ndarray,
    shots: int = 2000
) -> float:
    """
    Compute energy expectation value using shot-based sampling.
    
    This function mimics real quantum hardware by:
    1. Running the circuit multiple times (shots)
    2. Measuring in appropriate bases for each Pauli term
    3. Estimating expectation values from measurement statistics
    
    This introduces finite-shot sampling noise similar to what occurs
    on actual quantum devices, making it more realistic than exact
    statevector simulation.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized quantum circuit
    hamiltonian : SparsePauliOp
        Hamiltonian as sum of Pauli strings
    param_values : np.ndarray
        Parameter values to bind to the circuit
    shots : int, default=2000
        Number of measurement shots per Pauli term.
        More shots → better accuracy but slower execution.
        
    Returns
    -------
    float
        Estimated energy expectation value with shot noise
        
    Examples
    --------
    >>> energy_shots = energy_expectation_shots(ansatz, H, params, shots=5000)
    >>> # Compare with exact
    >>> energy_exact = energy_expectation(ansatz, H, params)
    >>> print(f"Shot noise: {abs(energy_shots - energy_exact):.6f}")
    
    Notes
    -----
    Measurement process:
    - For XX term: Apply H gates (X→Z basis), measure, compute parity
    - For YY term: Apply S†H gates (Y→Z basis), measure, compute parity
    - For ZZ term: Measure directly in computational basis, compute parity
    
    Parity-based expectation values:
    - Even parity (0 or 2 ones in bitstring) → eigenvalue +1
    - Odd parity (1 one in bitstring) → eigenvalue -1
    - ⟨P⟩ = (n_even - n_odd) / total_shots
    
    Trade-offs:
    - More shots: Better accuracy, longer runtime
    - Fewer shots: Faster but noisier, may affect optimization convergence
    """
    # Bind parameters to get concrete circuit
    bound_circuit = circuit.assign_parameters(param_values, inplace=False)
    
    # Initialize sampler (shot-based measurement simulator)
    sampler = StatevectorSampler()
    
    energy = 0.0
    
    # Decompose Hamiltonian into Pauli terms
    # Each term contributes: coefficient * ⟨Pauli_string⟩
    for pauli_string, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        # Create measurement circuit for this Pauli term
        meas_circuit = bound_circuit.copy()
        
        # Add basis rotations for each qubit based on Pauli operator
        # Qiskit orders qubits reversed in string representation
        for qubit_idx, pauli_op in enumerate(str(pauli_string)[::-1]):
            if pauli_op == 'X':
                # Rotate X basis to Z basis with Hadamard
                meas_circuit.h(qubit_idx)
            elif pauli_op == 'Y':
                # Rotate Y basis to Z basis with S†H
                meas_circuit.sdg(qubit_idx)
                meas_circuit.h(qubit_idx)
            # Z needs no rotation (already in computational basis)
        
        # Add measurements to all qubits
        meas_circuit.measure_all()
        
        # Run circuit with specified number of shots
        job = sampler.run([meas_circuit], shots=shots)
        result = job.result()
        
        # Get measurement counts from BitArray
        counts = result[0].data.meas.get_counts()
        
        # Compute expectation value from measurement statistics
        # For Pauli strings, eigenvalues are ±1 based on parity
        expectation = 0.0
        for bitstring, count in counts.items():
            # Count number of 1's in the bitstring
            num_ones = bitstring.count('1')
            parity = num_ones % 2
            
            # Even parity → +1, odd parity → -1
            eigenvalue = 1 if parity == 0 else -1
            expectation += count * eigenvalue
        
        # Normalize by total shots
        expectation /= shots
        
        # Add weighted contribution to total energy
        energy += float(np.real(coeff)) * expectation
    
    return energy


def energy_expectation_hardware(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    param_values: np.ndarray,
    backend,
    shots: int = 1024
) -> float:
    """
    Compute energy expectation value using REAL quantum hardware.
    
    This function executes the circuit on actual quantum processors
    via IBM Quantum, incorporating all physical noise sources:
    - Gate errors (imperfect unitary operations)
    - Decoherence (T1 and T2 relaxation times)
    - Readout errors (measurement bit-flip errors)
    - Crosstalk between qubits
    
    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized quantum circuit (should be pre-transpiled for hardware)
    hamiltonian : SparsePauliOp
        Hamiltonian operator (should be transpiled to match physical qubits)
    param_values : np.ndarray
        Parameter values to evaluate
    backend : IBMBackend
        The quantum computer backend to use
    shots : int, default=1024
        Number of measurement shots. Hardware has limited runtime,
        so fewer shots are typically used than in simulation.
        
    Returns
    -------
    float
        Measured energy expectation value from real hardware,
        including all physical noise and errors.
        
    Examples
    --------
    >>> from qiskit_ibm_runtime import QiskitRuntimeService
    >>> from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    >>> 
    >>> # Connect to IBM Quantum
    >>> service = QiskitRuntimeService(channel="ibm_quantum_platform")
    >>> backend = service.least_busy(simulator=False, operational=True)
    >>> 
    >>> # Transpile circuit and Hamiltonian for hardware
    >>> pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    >>> transpiled_ansatz = pm.run(ansatz)
    >>> # ... transpile Hamiltonian to match physical qubits ...
    >>> 
    >>> # Evaluate energy on hardware
    >>> energy_hw = energy_expectation_hardware(
    ...     transpiled_ansatz, H_transpiled, params, backend, shots=1024
    ... )
    
    Notes
    -----
    - Circuit must be transpiled to match hardware topology and native gates
    - Hamiltonian must be expanded to full hardware qubit count
    - Uses resilience_level=1 for basic error mitigation
    - Each call submits a job to IBM Quantum queue
    - Jobs may take minutes to hours depending on queue length
    - Monitor usage via IBM Quantum dashboard
    
    Error mitigation:
    - resilience_level=0: No mitigation (fastest but noisiest)
    - resilience_level=1: Basic mitigation (twirling, ZNE)
    - resilience_level=2: Advanced mitigation (PEC, slower)
    """
    # Create estimator with hardware backend
    # Pass backend as positional argument (EstimatorV2 API requirement)
    estimator = RuntimeEstimator(backend)
    
    # Set options for execution
    estimator.options.default_shots = shots
    estimator.options.resilience_level = 1  # Enable basic error mitigation
    
    # Bind parameters to get concrete circuit
    bound_circuit = circuit.assign_parameters(param_values)
    
    # Submit job to quantum hardware
    # Creates a PUB (Primitive Unified Bloc) - Qiskit Runtime API format
    job = estimator.run([(bound_circuit, hamiltonian)])
    
    # Wait for job to complete and retrieve result
    # This blocks until the job finishes (may take time due to queue)
    result = job.result()
    
    # Extract energy value
    energy = result[0].data.evs
    
    return float(np.real(energy))
