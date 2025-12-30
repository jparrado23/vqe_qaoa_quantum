"""
Transpilation utilities for quantum hardware execution.

Functions for transpiling quantum circuits and Hamiltonians for execution
on real quantum processors.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def transpile_for_hardware(
    circuit: QuantumCircuit,
    backend,
    optimization_level: int = 3
):
    """
    Transpile a quantum circuit for execution on specific hardware.
    
    Transpilation adapts the circuit to hardware constraints:
    - Maps logical qubits to physical qubits
    - Decomposes gates into native gate set
    - Optimizes circuit depth and gate count
    - Respects qubit connectivity topology
    
    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit to transpile
    backend : IBMBackend
        The target quantum computer backend
    optimization_level : int, default=3
        Optimization level (0-3):
        - 0: No optimization (fastest compilation)
        - 1: Light optimization
        - 2: Medium optimization
        - 3: Heavy optimization (best circuit quality, slowest)
        
    Returns
    -------
    QuantumCircuit
        Transpiled circuit ready for hardware execution
    dict
        Information about the transpilation:
        - 'physical_qubits': Physical qubit indices used
        - 'original_depth': Circuit depth before transpilation
        - 'transpiled_depth': Circuit depth after transpilation
        - 'gate_count_original': Gate count before
        - 'gate_count_transpiled': Gate count after
        
    Examples
    --------
    >>> from qiskit_ibm_runtime import QiskitRuntimeService
    >>> service = QiskitRuntimeService(channel="ibm_quantum_platform")
    >>> backend = service.least_busy(simulator=False, operational=True)
    >>> 
    >>> transpiled_circuit, info = transpile_for_hardware(ansatz, backend)
    >>> print(f"Mapped to qubits: {info['physical_qubits']}")
    >>> print(f"Depth: {info['original_depth']} → {info['transpiled_depth']}")
    """
    # Store original circuit stats
    original_depth = circuit.depth()
    original_gates = circuit.count_ops()
    
    # Create pass manager for transpilation
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend
    )
    
    # Transpile the circuit
    transpiled = pm.run(circuit)
    
    # Extract physical qubit mapping
    # BUG FIX: Must iterate through layout to find circuit qubits (register "q")
    # not ancilla qubits. initial_layout[i] looks up physical qubit i, not logical qubit i!
    initial_layout = transpiled.layout.initial_layout
    num_logical_qubits = circuit.num_qubits
    physical_qubits = []
    
    # Iterate over all physical qubits to find which ones map to circuit qubits
    for physical_idx in range(backend.num_qubits):
        try:
            virtual_qubit = initial_layout[physical_idx]
            # Check if this is a circuit qubit (register "q", not "ancilla")
            if hasattr(virtual_qubit, '_register') and virtual_qubit._register.name == "q":
                physical_qubits.append(physical_idx)
        except KeyError:
            # Physical qubit not used in layout
            continue
    
    # Sort by virtual qubit index to maintain correct order [logical 0, logical 1, ...]
    physical_qubits.sort(key=lambda p: initial_layout[p]._index)
    
    # Prepare information dictionary
    info = {
        'physical_qubits': physical_qubits,
        'original_depth': original_depth,
        'transpiled_depth': transpiled.depth(),
        'gate_count_original': original_gates,
        'gate_count_transpiled': transpiled.count_ops()
    }
    
    return transpiled, info


def transpile_hamiltonian(
    hamiltonian: SparsePauliOp,
    physical_qubits: list,
    num_qubits_hardware: int
) -> SparsePauliOp:
    """
    Expand a Hamiltonian to match hardware qubit count.
    
    When a circuit is transpiled for hardware, it gets mapped to specific
    physical qubits. The Hamiltonian must be expanded accordingly, inserting
    identity operators on all other qubits.
    
    Example:
        Original: "XX" on logical qubits [0, 1]
        Mapped to physical qubits [131, 132] on 156-qubit hardware
        Result: "II...IXXI...II" (156 qubits, X at positions 131 and 132)
    
    Parameters
    ----------
    hamiltonian : SparsePauliOp
        The original Hamiltonian (defined on logical qubits)
    physical_qubits : list[int]
        Physical qubit indices where logical qubits are mapped.
        Length must match the number of qubits in the Hamiltonian.
    num_qubits_hardware : int
        Total number of qubits in the quantum processor
        
    Returns
    -------
    SparsePauliOp
        Expanded Hamiltonian operating on full hardware qubit space
        
    Examples
    --------
    >>> # Original 2-qubit Hamiltonian
    >>> H = heisenberg_xxz_2q(J=1.0, delta=1.0)
    >>> 
    >>> # After transpiling circuit, logical qubits mapped to [131, 132]
    >>> physical_qubits = [131, 132]
    >>> 
    >>> # Expand Hamiltonian to 156-qubit hardware
    >>> H_transpiled = transpile_hamiltonian(H, physical_qubits, 156)
    >>> 
    >>> # Now H_transpiled has 156-qubit Pauli strings
    >>> print(f"Hamiltonian size: {len(next(iter(H_transpiled.paulis)))} qubits")
    
    Notes
    -----
    - Qiskit uses reversed qubit ordering in Pauli strings
    - Identity operators are added on all unmapped qubits
    - Coefficients remain unchanged
    - Essential for hardware execution with RuntimeEstimator
    """
    new_paulis = []
    new_coeffs = []
    
    for pauli_str, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        # Create identity string for all hardware qubits
        expanded_str = ['I'] * num_qubits_hardware
        
        # Map each Pauli operator to its physical qubit location
        # Note: Qiskit reverses qubit order in string representation
        for logical_idx, pauli_op in enumerate(str(pauli_str)[::-1]):
            physical_idx = physical_qubits[logical_idx]
            expanded_str[physical_idx] = pauli_op
        
        # Reverse back to Qiskit convention
        expanded_pauli_str = ''.join(expanded_str[::-1])
        new_paulis.append(expanded_pauli_str)
        new_coeffs.append(coeff)
    
    return SparsePauliOp(new_paulis, coeffs=np.array(new_coeffs))
